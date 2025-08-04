#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse
import time
import getpass
import os
import subprocess
from typing import Optional
import socket
from select import select

import torch
import paramiko
import pandas as pd

import config
from test_read_benchmark import (
    benchmark_csv,
    result_csv,
    logger,
    DATASETS,
    init_result_csv_index,
    create_and_save_params,
)
from test_msprof_npu_hstu import msprof_main


INDEX_STR = "index"
GPU_FW_TIME = "gpu_fw_time"
NPU_FW_TIME = "npu_fw_time"
NPU_BW_TIME = "npu_bw_time"
GPU_BW_TIME = "gpu_bw_time"


_remote_password_cache = None


def get_remote_password() -> Optional[str]:
    """Securely retrieves and caches the GPU server password (single input, reusable for the session).

    Returns:
        Optional[str]: The password if successfully retrieved, None otherwise.
    """
    global _remote_password_cache

    if _remote_password_cache is not None:
        return _remote_password_cache

    try:
        # Interactive input (hidden)
        logger.info("\n[Security Notice] Password input will not display characters.")
        password = getpass.getpass(
            prompt="Enter GPU server password (valid for this session): "
        )
        if password.strip():
            _remote_password_cache = password
            return _remote_password_cache
        else:
            logger.error("Password cannot be empty.")
            return None

    except Exception as e:
        logger.error(f"Password retrieval failed: {str(e)}")
        return None


def execute_remote_linux_cmd(client, cmd, timeout=600):
    """Secure execution of Linux remote commands (supports timeout control and complete output capture)"""
    stdin, stdout, stderr = None, None, None
    exit_status = -1
    output_buffer = []
    error_buffer = []

    try:
        # Create execution channel (set timeout to prevent network blocking)
        transport = client.get_transport()
        channel = transport.open_session(timeout=timeout)
        channel.settimeout(timeout)

        # Execute command (non-blocking mode)
        channel.exec_command(cmd)
        stdin = channel.makefile_stdin("wb")
        stdout = channel.makefile("r")
        stderr = channel.makefile_stderr("r")

        # Use select for efficient I/O multiplexing
        start_time = time.time()
        while not channel.exit_status_ready():
            # Timeout check
            if time.time() - start_time > timeout:
                raise socket.timeout(f"Command timeout after {timeout}s: {cmd}")

            # Wait for readable events (0.5s polling interval)
            rlist, _, _ = select([channel], [], [], 0.5)
            if not rlist:
                continue

            # Prioritize reading error stream (avoid buffer overflow)
            while channel.recv_stderr_ready():
                error_buffer.append(stderr.read(4096))

            while channel.recv_ready():
                output_buffer.append(stdout.read(4096))

        # Get final exit status
        exit_status = channel.recv_exit_status()

        # Read all remaining output
        output_buffer.append(stdout.read())
        error_buffer.append(stderr.read())

    except socket.timeout as te:
        logger.error(f"Command execution timed out: {te}")
        # Attempt to terminate remote process (send SIGTERM)
        if "channel" in locals():
            channel.close()
        return False
    except Exception as e:
        logger.error(f"SSH command execution failed: {e}", exc_info=True)
        return False
    finally:
        # Safely close channels (without closing underlying client connection)
        for stream in [stdin, stdout, stderr]:
            try:
                if stream:
                    stream.close()
            except Exception as e:
                logger.info(e)

        try:
            if "channel" in locals():
                channel.close()
        except Exception as e:
            logger.info(e)

    # Build output results
    decoded_output = [
        item.decode("utf-8") if isinstance(item, bytes) else item
        for item in output_buffer
    ]
    decoded_error = [
        item.decode("utf-8") if isinstance(item, bytes) else item
        for item in error_buffer
    ]
    full_output = "".join(decoded_output)
    full_error = "".join(decoded_error)

    # Log results (limit log length)
    logger.info(f"Command [{cmd}] exited with status: {exit_status}")
    if full_output:
        logger.debug(
            f"stdout: {full_output[:2000]}{'...' if len(full_output)>2000 else ''}"
        )
    if full_error:
        logger.warning(
            f"stderr: {full_error[:2000]}{'...' if len(full_error)>2000 else ''}"
        )

    return exit_status == 0


def transfer_and_execute(bx):
    if not isinstance(bx, int):
        raise ValueError(f"input must be an integer but got {bx}")
    remote_host = config.GPU_IP
    remote_user = config.GPU_USER
    remote_password = get_remote_password()
    remote_dir = os.path.realpath(config.RECSYS_DIR)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cmd = f"cd {remote_dir} && {os.path.realpath(config.PYTHON3)} test_gpu_hstu.py --index={bx}"

    try:
        client.connect(
            remote_host, port=22, username=remote_user, password=remote_password
        )
        sftp = client.open_sftp()
        files_to_transfer = ["test_gpu_hstu.py", "test_read_benchmark.py", "config.py"]
        for file in files_to_transfer:
            sftp.put(file, os.path.join(remote_dir, file))
        sftp.close()

        logger.info(f"Executing remote script: {cmd}")
        return execute_remote_linux_cmd(client, cmd)
    except Exception as e:
        logger.error(f"Failed to execute remote script: {cmd}")
        logger.error(e)
        return False
    finally:
        # 安全关闭连接
        try:
            if client:
                client.close()
        except Exception as close_error:
            logger.error(f"SSH close error: {close_error}")
        logger.info(f"Exit remote script: {cmd}")


def compare_npu_gpu_precision(save_dir=DATASETS, device="cpu"):
    logger.info(f"Compating npu and gpu results of {save_dir}")
    try:
        data_type = torch.load(os.path.join(save_dir, "dtype.pth"), map_location=device)
        npu_out = torch.load(
            os.path.join(save_dir, "npu_out.pth"), map_location=device
        ).to(dtype=data_type)
        gpu_out = torch.load(
            os.path.join(save_dir, "gpu_out.pth"), map_location=device
        ).to(dtype=data_type)
        gpu_out = gpu_out.view(gpu_out.shape[0], -1)

        npu_q = torch.load(os.path.join(save_dir, "npu_q.pth"), map_location=device).to(
            dtype=data_type
        )
        gpu_q = torch.load(os.path.join(save_dir, "gpu_q.pth"), map_location=device).to(
            dtype=data_type
        )

        npu_k = torch.load(os.path.join(save_dir, "npu_k.pth"), map_location=device).to(
            dtype=data_type
        )
        gpu_k = torch.load(os.path.join(save_dir, "gpu_k.pth"), map_location=device).to(
            dtype=data_type
        )

        npu_v = torch.load(os.path.join(save_dir, "npu_v.pth"), map_location=device).to(
            dtype=data_type
        )
        gpu_v = torch.load(os.path.join(save_dir, "gpu_v.pth"), map_location=device).to(
            dtype=data_type
        )

    except Exception as e:
        logger.error(f"error : {e}")
        return False

    if data_type == torch.bfloat16:
        eps = 1e-2
    elif data_type == torch.float16:
        eps = 1e-3
    else:
        logger.error(f"error type : {data_type}")
        return False

    try:
        out_close = torch.allclose(npu_out, gpu_out, eps, eps)
        out_q = torch.allclose(npu_q, gpu_q, eps, eps)
        out_k = torch.allclose(npu_k, gpu_k, eps, eps)
        out_v = torch.allclose(npu_v, gpu_v, eps, eps)
    except Exception as e:
        logger.error(f"error : {e}")
        return False

    logger.info(f"npu_out vs gpu_out: {out_close}")
    logger.info(f"npu_q vs gpu_q: {out_q}")
    logger.info(f"npu_k vs gpu_k: {out_k}")
    logger.info(f"npu_v vs gpu_v: {out_v}")

    ret = out_close and out_q and out_k and out_v
    return ret


def update_index_csv(df_res, benchmark_df, bx, precision):
    mask_res = df_res[INDEX_STR] == bx
    mask_benchmark = benchmark_df[INDEX_STR] == bx

    df_res.loc[mask_res, "precision"] = precision
    df_res.loc[mask_res, "npu_fw/gpu_fw"] = (
        df_res.loc[mask_res, GPU_FW_TIME] / df_res.loc[mask_res, NPU_FW_TIME]
    )
    df_res.loc[mask_res, "npu_bw/gpu_bw"] = (
        df_res.loc[mask_res, GPU_BW_TIME] / df_res.loc[mask_res, NPU_BW_TIME]
    )

    df_res.loc[mask_res, "npu_fw+bw/gpu_fw+bw"] = df_res.loc[
        mask_res, [GPU_FW_TIME, GPU_BW_TIME]
    ].sum(axis=1) / df_res.loc[mask_res, [NPU_FW_TIME, NPU_BW_TIME]].sum(axis=1)
    df_res.loc[mask_res, "npu_fw/benchmark"] = (
        benchmark_df.loc[mask_benchmark, NPU_FW_TIME].item()
        / df_res.loc[mask_res, NPU_FW_TIME].item()
    )
    df_res.loc[mask_res, "npu_bw/benchmark"] = (
        benchmark_df.loc[mask_benchmark, NPU_BW_TIME].item()
        / df_res.loc[mask_res, NPU_BW_TIME].item()
    )
    df_res.loc[mask_res, "npu_fw+bw/benchmark"] = benchmark_df.loc[
        mask_benchmark, [NPU_FW_TIME, NPU_BW_TIME]
    ].sum(axis=1).item() / df_res.loc[mask_res, [NPU_FW_TIME, NPU_BW_TIME]].sum(axis=1).item()



def retry_operation(operation, operation_name, bx, max_retries=2):
    for attempt in range(max_retries):
        ret = operation(bx)
        if ret:
            return True
        logger.warning(
            f"{operation_name} failed for benchmark {bx}, attempt {attempt + 1}/{max_retries}"
        )
    logger.error(
        f"{operation_name} failed for benchmark {bx} after {max_retries} retries"
    )
    return False


def main(index=None):
    benchmark_df = pd.read_csv(benchmark_csv)
    benchmark_df[INDEX_STR] = benchmark_df[INDEX_STR].astype(int)
    all_indices = benchmark_df[INDEX_STR].tolist()

    if index is not None:
        all_indices = [index]
    failed = []
    for bx in all_indices:
        logger.info(f"benchmark {bx} testing")
        init_result_csv_index(bx)

        df_res = pd.read_csv(result_csv)

        if (
            bx in df_res[INDEX_STR].values
            and df_res[df_res[INDEX_STR] == bx].notna().all().all()
        ):
            logger.info(f"benchmark {bx} already, pass")
            continue

        create_and_save_params(bx)
        remote_success = retry_operation(transfer_and_execute, "Remote execution", bx)
        if not remote_success:
            failed.append(bx)
            continue

        local_success = retry_operation(msprof_main, "Local execution", bx)
        if not local_success:
            failed.append(bx)
            continue

        df_res = pd.read_csv(result_csv)
        precision = compare_npu_gpu_precision()
        update_index_csv(df_res, benchmark_df, bx, precision)
        df_res.to_csv(result_csv, index=False)
        logger.info(f"benchmark {bx} tested")

    if len(failed) != 0:
        logger.error("index %s failed!", ", ".join(str(x) for x in failed))
    else:
        logger.info("All successed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark tests.")
    parser.add_argument("--index", type=int, help="benchmark to run. Default run all.")
    args = parser.parse_args()
    main(args.index)

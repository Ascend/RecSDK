#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.
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
import ast
import logging
import math
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s %(lineno)d [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("test_benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

matplotlib.use("Agg")
INDEX_STR = "index"
DATASETS = os.path.join(os.path.realpath(config.NFS_DIR), "datasets")
benchmark_csv = os.path.join(os.path.realpath(config.NFS_DIR), "benchmark.csv")
result_csv = os.path.join(os.path.realpath(config.NFS_DIR), "result.csv")

column_names = [
    "index",
    "shape_info",
    "total_len",
    "batch_size",
    "heads",
    "heads_rab",
    "max_seq_len_q",
    "max_seq_len_k",
    "max_context_len",
    "max_target_len",
    "target_group_size",
    "attn_dim",
    "hidden_dim",
    "alpha",
    "has_rab",
    "has_drab",
    "window_size",
    "run_benchmark",
    "dtype",
    "full_batch",
    "is_delta_q",
    "format",
    "npu_fw_time",
    "npu_bw_time",
    "gpu_fw_time",
    "gpu_bw_time",
    "precision",
    "npu_fw/gpu_fw",
    "npu_bw/gpu_bw",
    "npu_fw+bw/gpu_fw+bw",
    "npu_fw/benchmark",
    "npu_bw/benchmark",
    "npu_fw+bw/benchmark",
]

column_left = column_names[: column_names.index("format") + 1]

hstu_required_params = {
    "total_len": int,
    "batch_size": int,
    "heads": int,
    "heads_rab": int,
    "max_seq_len_q": int,
    "max_seq_len_k": int,
    "max_context_len": int,
    "max_target_len": int,
    "target_group_size": int,
    "attn_dim": int,
    "hidden_dim": int,
    "alpha": float,
    "has_rab": bool,
    "has_drab": bool,
    "window_size": tuple,
    "run_benchmark": int,
    "dtype": torch.dtype,
    "full_batch": bool,
    "is_delta_q": bool,
}

generate_params = [
    "total_len",
    "batch_size",
    "heads",
    "heads_rab",
    "max_seq_len_q",
    "max_seq_len_k",
    "max_context_len",
    "max_target_len",
    "target_group_size",
    "attn_dim",
    "hidden_dim",
    "window_size",
    "dtype",
    "full_batch",
    "has_drab",
    "is_delta_q",
]


def convert_value(value, required_type):
    if isinstance(value, str):
        if value == "torch.float16":
            return torch.float16
        elif value == "torch.bfloat16":
            return torch.bfloat16
        else:
            try:
                value = ast.literal_eval(value)
            except ValueError:
                pass
    try:
        return required_type(value)
    except (ValueError, TypeError):
        return value


def read_and_validate_parameters(index, csv_file_path=benchmark_csv):
    try:
        # Try to read CSV file
        df_benchmark = pd.read_csv(csv_file_path, encoding="utf-8")

        # Check if index column exists
        if INDEX_STR not in df_benchmark.columns:
            logger.error("Missing index column in CSV: %s", INDEX_STR)
            return None, None

        # Filter rows by index
        df_benchmark = df_benchmark.loc[df_benchmark[INDEX_STR] == index]
        if df_benchmark.empty:
            logger.info("No data found for index %d", index)
            return None, None

        # Extract required parameters
        try:
            row = df_benchmark[list(hstu_required_params.keys())]
        except KeyError as e:
            logger.error("Missing required columns in CSV: %s", e)
            return None, None

        # Convert to dict and validate types
        params = row.iloc[0].to_dict()
        for key, required_type in hstu_required_params.items():
            try:
                params[key] = convert_value(params[key], required_type)
            except (ValueError, TypeError) as e:
                logger.error("Type conversion failed for parameter %s: %s", key, e)
                return None, None

        logger.info("Parameters for index %d: %s", index, params)
        return df_benchmark, params

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)  # Added exc_info for stack trace
        return None, None


def init_result_csv_index(index):
    try:
        # Read benchmark CSV and ensure index column is integer type
        ben_df = pd.read_csv(benchmark_csv)
        ben_df[INDEX_STR] = ben_df[INDEX_STR].astype(int)

        # Create empty result file if not exists
        if not os.path.exists(result_csv):
            pd.DataFrame(columns=column_names).to_csv(result_csv, index=False)

        # Read result file
        df_res = pd.read_csv(result_csv)

        # Check if index exists in benchmark data
        benchmark_row = ben_df.loc[ben_df[INDEX_STR] == index, column_left]
        if benchmark_row.empty:  # Explicit empty case handling
            logger.warning("Index %d not found in benchmark file, skipping", index)
            raise ValueError("Row %d not found in %s" % (index, benchmark_csv))

        # Check if index already exists in result file
        if index in df_res[INDEX_STR].values:
            return

        # Add new row and save
        new_row = pd.DataFrame([benchmark_row.iloc[0]], columns=column_left)
        df_res = pd.concat([df_res, new_row], ignore_index=True)
        df_res.to_csv(result_csv, index=False)
        logger.info("Successfully added index %d to result file %s", index, result_csv)

    except Exception as e:
        logger.error("Error initializing result file: %s", str(e), exc_info=True)
        raise


def construct_mask(
    seqlen_c,
    seqlen,
    seqlen_t=0,
    target_group_size=1,
    window_size=(-1, -1),  # -1 means infinite window size
    seq_offsets=None,
    num_contexts=None,
    device=None,
):
    seqlen = seqlen_c + seqlen + seqlen_t
    bs = seq_offsets.size(0) - 1

    mask = torch.zeros((seqlen, seqlen), device=device, dtype=torch.bool)
    if window_size[0] < 0 and window_size[1] == 0:
        # causal mask
        for i in range(seqlen):
            mask[i, : i + 1] = True

        # context mask
        if seqlen_c != 0:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
                ).item()
                mask[i, 0, : num_contexts[i], :target_start] = True

        # target mask
        if seqlen_t != 0:
            mask = (
                mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
                if mask.ndim == 2
                else mask
            )
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
                ).item()
                # target group mask
                if target_group_size > 1:
                    group_num = math.ceil((seqlen - target_start) / target_group_size)
                    for j in range(group_num):
                        for k in range(
                            min(
                                target_group_size,
                                seqlen - target_start - j * target_group_size,
                            )
                        ):
                            mask[
                                i,
                                0,
                                target_start + j * target_group_size + k,
                                target_start: target_start + j * target_group_size,
                            ] = False
                else:
                    for j in range(target_start, seqlen):
                        mask[i, 0, j, target_start:j] = False

    # local mask
    else:
        window_size_0 = window_size[0] if window_size[0] > 0 else seqlen
        window_size_1 = window_size[1] if window_size[1] > 0 else seqlen
        for i in range(seqlen):
            mask[i, max(0, i - window_size_0): min(seqlen, i + window_size_1 + 1)] = (
                True
            )
    return mask


def gen_seq(length, max_value, total_sum):
    if max_value * length < total_sum:
        raise ValueError("total_sum error %d" % total_sum)
    logger.info("gen_seq with total_sum %d", total_sum)
    if length == 1:
        return np.array([total_sum])
    if length == 2:
        return np.array([max_value, total_sum - max_value])
    remaining_sum = total_sum - max_value
    mean_value = remaining_sum // (length - 2)
    min_val = remaining_sum - mean_value * (length - 2)
    sequence = [mean_value] * (length - 2)
    if min_val == 0 and sequence[-1] > 1:
        min_val += 1
        sequence[-1] -= 1
    sequence.extend([max_value, min_val])
    return np.array(sequence)


def adjust_ratio(total_sum, max_context_len, max_seq_len_k, max_target_len):
    """Adjust ratio distribution based on given parameters."""
    # Initialize result variables
    total_content, total_k, total_target = 0, 0, 0

    # Check denominator
    denominator = max_context_len + max_seq_len_k + max_target_len
    if denominator == 0:
        logger.debug("All max lengths are 0, returning zeros")
        return 0, 0, 0  # Return zeros if all inputs are zero

    # Handle zero cases
    if max_context_len == 0:
        total_content = 0
    if max_seq_len_k == 0:
        total_k = 0
    if max_target_len == 0:
        total_target = 0

    # Calculate remaining sum to distribute
    remaining_sum = total_sum - (total_content + total_k + total_target)
    if remaining_sum == 0:
        logger.debug("No remaining sum to distribute")
        return total_content, total_k, total_target

    # Calculate valid denominator (excluding zero terms)
    valid_denominator = 0
    if max_context_len > 0:
        valid_denominator += max_context_len
    if max_seq_len_k > 0:
        valid_denominator += max_seq_len_k
    if max_target_len > 0:
        valid_denominator += max_target_len

    if valid_denominator == 0:
        raise ValueError("valid_denominator cannot be 0")

    logger.debug("Distributing remaining sum %d with valid denominator %d", remaining_sum, valid_denominator)

    try:
        # Distribute remaining sum proportionally
        if max_context_len > 0 and valid_denominator != 0:
            total_content += int(round(remaining_sum * max_context_len / valid_denominator))

        if max_seq_len_k > 0 and valid_denominator != 0:
            total_k += int(round(remaining_sum * max_seq_len_k / valid_denominator))

        if max_target_len > 0 and valid_denominator != 0:
            total_target += int(round(remaining_sum * max_target_len / valid_denominator))
    except ZeroDivisionError as e:
        logger.info(e)
        raise e

    # Handle rounding errors
    diff = total_sum - (total_content + total_k + total_target)
    if diff != 0:
        logger.debug("Adjusting for rounding difference of %d", diff)
        total_target += diff  # Default adjustment to target

    logger.info("Final distribution: total_k=%d, total_content=%d, total_target=%d", \
                total_k, total_content, total_target)
    return total_k, total_content, total_target


def generate_input(
    total_len: int,
    batch_size: int,
    heads: int,
    heads_rab: Optional[int],
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    window_size: Tuple[int, int],
    dtype: torch.dtype,
    full_batch: bool,
    has_drab: bool,
    is_delta_q: bool,
):
    device_str = "cpu"
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    target_group_size > 1
    
    # Modification Note: Original random length generation caused uncontrolled total length,
    # leading to unstable computational load. Now using proportional allocation based on total_len.
    logger.info("generate with total_len %d", total_len)
    
    # Allocate total length proportionally according to max_seq_len_k/max_context_len/max_target_len
    total_k, total_content, total_target = adjust_ratio(
        total_len, max_context_len, max_seq_len_k, max_target_len
    )
    
    # Generate key sequence lengths (proportionally allocated)
    lengths_k = (
        torch.from_numpy(gen_seq(batch_size, max_seq_len_k, total_k))
        .to(device_str)
        .int()
    )
    
    # Generate context sequence lengths (proportionally allocated)
    num_contexts = (
        torch.from_numpy(gen_seq(batch_size, max_context_len, total_content))
        .to(device_str)
        .int()
    )
    
    # Generate target sequence lengths (proportionally allocated)
    num_targets = (
        torch.from_numpy(gen_seq(batch_size, max_target_len, total_target))
        .to(device_str)
        .int()
    )

    # Generate lengths for context

    seq_offsets_c = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_c[1:] = torch.cumsum(num_contexts, dim=0)

    # Generate lengths for historial qkv

    seq_offsets_k = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_k[1:] = torch.cumsum(lengths_k, dim=0)

    # Generate lengths for target qkv
    seq_offsets_t = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_t[1:] = torch.cumsum(num_targets, dim=0)

    # Generate lengths for delta q
    if is_delta_q:
        if full_batch:
            lengths_q = (
                torch.ones(
                    (batch_size,), device=torch.device(device_str), dtype=torch.int32
                )
                * max_seq_len_q
            )
        else:
            # lengths_q[i] is an integer between 1 and min(max_seq_len_q, lengths_k[i])
            lengths_q = torch.zeros(
                (batch_size,), device=torch.device(device_str), dtype=torch.int32
            )
            for i in range(batch_size):
                lengths_q[i] = torch.randint(
                    1,
                    min(max_seq_len_q, lengths_k[i]) + 1,
                    size=(1,),
                    device=torch.device(device_str),
                )
        seq_offsets_q = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
        )
        seq_offsets_q[1:] = torch.cumsum(lengths_q, dim=0)
    else:
        seq_offsets_q = seq_offsets_k

    # Lengths for whole q, kv
    seq_offsets_q_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_q_wt = seq_offsets_c + seq_offsets_q + seq_offsets_t
    seq_offsets_k_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_k_wt = seq_offsets_c + seq_offsets_k + seq_offsets_t

    l_q = int(seq_offsets_q_wt[-1].item())
    l_k = int(seq_offsets_k_wt[-1].item())
    if dtype == torch.float8_e4m3fn:
        dtype_init = torch.float16
    else:
        dtype_init = dtype

    # Generate q, k, v for history + target
    q = (
        torch.empty(
            (l_q, heads, attn_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    k = (
        torch.empty(
            (l_k, heads, attn_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    v = (
        torch.empty(
            (l_k, heads, hidden_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    rab = None
    if has_drab:
        rab = torch.empty(
            (
                batch_size,
                heads if heads_rab is None else heads_rab,
                max_context_len + max_seq_len_k + max_target_len,
                max_context_len + max_seq_len_k + max_target_len,
            ),
            dtype=dtype_init,
            device=torch.device(device_str),
        ).uniform_(-1, 1)
        rab = rab.requires_grad_()

    if window_size[0] == -1 and window_size[1] == -1:
        attn_mask = None
    else:
        attn_mask = (
            construct_mask(
                seqlen_c=max_context_len,
                seqlen=max_seq_len_k,
                seqlen_t=max_target_len,
                target_group_size=target_group_size,
                window_size=window_size,
                num_contexts=num_contexts,
                seq_offsets=seq_offsets_k,
            )
            .cpu()
            .to(torch.float32)
        )
    grad = torch.rand_like(v)
    params = {
        "l_q": l_q,
        "l_k": l_k,
        "num_contexts": num_contexts if has_context else None,
        "seq_offsets_q_wt": seq_offsets_q_wt,
        "seq_offsets_k_wt": seq_offsets_k_wt,
        "num_targets": num_targets if has_target else None,
        "q": q,
        "k": k,
        "v": v,
        "rab": rab,
        "attn_mask": attn_mask,
        "grad": grad,
    }
    return params


def save_mask(matrix, title="matrix"):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "CustomMap", [(1, 1, 1), (0.8, 0.902, 0.8)], N=256
    )

    max_pixels = 65536
    dpi = 100

    width_inches = min(12, max(6, matrix.shape[1] * 0.05))
    height_inches = min(12, max(6, matrix.shape[0] * 0.05))

    width_pixels = width_inches * dpi
    height_pixels = height_inches * dpi

    if width_pixels > max_pixels or height_pixels > max_pixels:
        scale_factor = min(max_pixels / width_pixels, max_pixels / height_pixels)
        width_inches *= scale_factor
        height_inches *= scale_factor

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    img = ax.imshow(
        matrix, cmap=cmap, origin="lower", vmin=0, vmax=1, interpolation="nearest"
    )

    ax.xaxis.set_ticks_position("top")
    ax.yaxis.set_ticks_position("left")
    ax.invert_yaxis()

    plt.colorbar(img)
    plt.title(title)
    plt.xlabel("seq_len")
    plt.ylabel("seq_len")
    plt.savefig(f"{title}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info(f"save {title}.png")


PARAM_META = {
    # Tensor parameters (name: (type, required, description))
    "l_q": ("Tensor", True, "Query sequence length"),
    "l_k": ("Tensor", True, "Key sequence length"),
    "num_contexts": ("Tensor", True, "Number of contexts"),
    "seq_offsets_q_wt": ("Tensor", True, "Query sequence weight offsets"),
    "seq_offsets_k_wt": ("Tensor", True, "Key sequence weight offsets"),
    "num_targets": ("Tensor", True, "Number of targets"),
    "q": ("Tensor", True, "Query tensor"),
    "k": ("Tensor", True, "Key tensor"),
    "v": ("Tensor", True, "Value tensor"),
    "rab": ("Tensor", True, "Relative attention bias"),
    "attn_mask": ("Tensor", True, "Attention mask"),
    "grad": ("Tensor", True, "Gradient tensor"),
    # Configuration parameters
    "dtype": ("Config", True, "Data type"),
    "max_context_len": ("Config", True, "Max context length"),
    "max_seq_len_q": ("Config", True, "Max query sequence length"),
    "max_target_len": ("Config", True, "Max target length"),
    "alpha": ("Config", True, "silu scale tensor"),
}



def _get_save_paths(save_dir: str) -> Dict[str, str]:
    return {
        name: os.path.normpath(os.path.join(save_dir, f"{name}.pth"))
        for name in PARAM_META
        if not name.startswith("_")
    }


def save_params(save_dir=DATASETS, **kwargs):
    # Normalize path
    save_dir = os.path.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Validate required parameters
    missing = [
        name
        for name, (_, required, _) in PARAM_META.items()
        if required and name not in kwargs
    ]
    if missing:
        logger.error("Missing required parameters: %s", ", ".join(missing))
        raise ValueError("Required parameters missing: %s" % ", ".join(missing))

    # Get all save paths
    paths = _get_save_paths(save_dir)
    image_name = "image_name"
    # Save parameters with logging
    logger.info("\n%s", "=" * 50)
    logger.info("[SAVE] Target directory: %s", save_dir)
    for name, path in paths.items():
        if name in kwargs:
            torch.save(kwargs[name], path)
            param = kwargs[name]
            log_msg = "%s: " % name.ljust(15)
            if torch.is_tensor(param):
                log_msg += "shape=%s | dtype=%s" % (
                    str(param.shape).ljust(18),
                    param.dtype,
                )
            else:
                log_msg += str(param)
            logger.info("%s -> %s", log_msg, path)

    # Save attention matrix image if provided
    if "attn_mask" in kwargs and image_name in kwargs:
        mask = kwargs["attn_mask"]
        if mask.dim() == 4:
            save_matrix = mask[0, 0].cpu()
        elif mask.dim() == 3:
            save_matrix = mask[0].cpu()
        else:
            save_matrix = mask.cpu()

        img_path = os.path.join(save_dir, kwargs[image_name])
        save_mask(save_matrix, img_path)

    # Create completion flag
    flag_path = os.path.join(save_dir, "complete.flag")
    torch.save(torch.tensor(1), flag_path)
    logger.info("%s\n[SUCCESS] All parameters saved\n%s", "=" * 50, "=" * 50)


def load_params(save_dir=DATASETS, device="cpu") -> Dict[str, object]:
    save_dir = os.path.normpath(save_dir)
    paths = _get_save_paths(save_dir)

    # Validate directory integrity
    if not os.path.exists(paths["l_q"]):
        logger.error("Invalid parameter directory: %s", save_dir)
        raise FileNotFoundError("Invalid parameter directory: %s" % save_dir)

    # Load parameters with logging (lazy interpolation)
    logger.info("\n%s", "=" * 50)
    logger.info("[LOAD] Source directory: %s", save_dir)
    params = {}
    for name, path in paths.items():
        if os.path.exists(path):
            params[name] = torch.load(path, map_location=device)
            log_msg = "%s: " % name.ljust(15)
            if torch.is_tensor(params[name]):
                log_msg += "shape=%s | dtype=%s" % (
                    str(params[name].shape).ljust(18),
                    params[name].dtype,
                )
            else:
                log_msg += str(params[name])
            logger.info("%s <- %s", log_msg, path)

    logger.info("%s\n[SUCCESS] All parameters loaded\n%s", "=" * 50, "=" * 50)
    return params


def create_and_save_params(bx):
    df, bparams = read_and_validate_parameters(bx)
    image_name = "image_name"
    bparams[image_name] = f"{bx}_{df.shape_info.item()}"
    init_result_csv_index(bx)
    df_resg = pd.read_csv(result_csv)
    logger.info(df_resg[df_resg[INDEX_STR] == bx])
    gene_param_dict = {k: bparams[k] for k in generate_params}
    iparams = generate_input(**gene_param_dict)
    iparams[image_name] = bparams[image_name]

    extra_params = [
        "dtype",
        "max_context_len",
        "max_seq_len_q",
        "max_target_len",
        "alpha",
    ]
    for param_name in extra_params:
        if param_name not in iparams:
            iparams[param_name] = bparams[param_name]

    save_params(**iparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read CSV file and run a specific index benchmark"
    )
    parser.add_argument(
        "--index", type=int, required=True, help="index of the benchmark to run"
    )
    args = parser.parse_args()
    benchmark_df = pd.read_csv(benchmark_csv)
    benchmark_df[INDEX_STR] = benchmark_df[INDEX_STR].astype(int)
    all_indices = benchmark_df[INDEX_STR].tolist()
    if args.index:
        all_indices = [args.index]
    for bxx in all_indices:
        create_and_save_params(bxx)
        # Load example
        loaded = load_params()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import argparse
import glob
import subprocess
import pandas as pd

import config
from test_read_benchmark import (
    logger,
    read_and_validate_parameters,
    result_csv,
    init_result_csv_index,
)


def msprof_main(index):
    index_str = "index"
    try:
        # Initialize the result CSV file
        init_result_csv_index(index)
        df_res = pd.read_csv(result_csv)

        # Check if the current index has already been processed
        if (
            df_res.loc[df_res[index_str] == index, "npu_fw_time"].notna().all()
            and df_res.loc[df_res[index_str] == index, "npu_bw_time"].notna().all()
        ):
            logger.info(f"Benchmark with index {index} is already done. Exiting.")
            return True

        # Read and validate parameters
        _, params = read_and_validate_parameters(index)

        # Execute the msprof command
        cmd = f'rm -rf profnpu/ ; msprof --application="python3 test_npu_hstu.py --index={index}" --output=profnpu'
        ret = os.system(cmd)
        if ret != 0:
            logger.error(f"Command execution failed (ret={ret}): {cmd}")
            return False

        # Locate the generated CSV file
        search_dir = os.path.join(os.path.realpath(config.NFS_DIR), "profnpu")
        csv_files = glob.glob(
            f"{search_dir}/PROF_*/mindstudio_profiler_output/op_stati*.csv"
        )
        if len(csv_files) == 0:
            logger.error(f"MSProf run failed. No CSV file found in {search_dir}.")
            return False

        csv_file = csv_files[0]
        logger.info(f"Profile located at: {csv_file}")

        # Read the CSV file
        df_op_stati = pd.read_csv(csv_file)

        # Extract Forward and Backward data
        forward_row = df_op_stati[df_op_stati["OP Type"] == "HstuDenseForward"]
        backward_row = df_op_stati[df_op_stati["OP Type"] == "HstuDenseBackward"]

        # Check if the data exists
        if forward_row.empty or backward_row.empty:
            missing_ops = []
            if forward_row.empty:
                missing_ops.append("HstuDenseForward")
            if backward_row.empty:
                missing_ops.append("HstuDenseBackward")
            logger.error(f"Missing OP types in CSV: {', '.join(missing_ops)}")
            return False

        # Update the result DataFrame
        df_res.loc[df_res[index_str] == index, "npu_fw_time"] = (
            forward_row["Avg Time(us)"].squeeze() / 1000
        )
        df_res.loc[df_res[index_str] == index, "npu_bw_time"] = (
            backward_row["Avg Time(us)"].squeeze() / 1000
        )

        # Save the results
        df_res.to_csv(result_csv, index=False)

        # Log the results
        logger.info(
            f"Forward time: {df_res.loc[df_res[index_str] == index, 'npu_fw_time'].values[0]} ms"
        )
        logger.info(
            f"Backward time: {df_res.loc[df_res[index_str] == index, 'npu_bw_time'].values[0]} ms"
        )

        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty CSV file or invalid data: {e}")
        return False
    except KeyError as e:
        logger.error(f"Missing required column in DataFrame: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read CSV file and run a specific index benchmark"
    )
    parser.add_argument(
        "--index", type=int, required=True, help="index of the benchmark to run"
    )
    args = parser.parse_args()
    if msprof_main(args.index):
        logger.info("success")

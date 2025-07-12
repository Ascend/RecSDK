#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
# Description: performace analysis tool
# Author: MindSDK
# Create: 2023
# History: NA


curr_path=$(cd $(dirname $0); pwd)

# ---------------config start---------------------
model_run_path=/path/to/model/run
run_cmd="bash run.sh"
# ---------------config end---------------------

# ------------------------------+
#            msprof             +
# ------------------------------+
output_path="${model_run_path}"/msprof_out

cd "${model_run_path}"
rm -rf "${output_path}"

msprof --application="${run_cmd}" --output="${output_path}"

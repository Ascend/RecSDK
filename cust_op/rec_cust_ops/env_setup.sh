#!/bin/bash
# ============================================================================
# RecSDK NPU Ops Environment Setup
# ============================================================================
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

# 获取本脚本所在的目录 (即安装后的 rec_cust_ops 包路径)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 动态探测并拼接所有的芯片级自定义算子路径 (如 custom_opp/A2, custom_opp/A5 等)
EXTRA_OPP_PATHS=""
for chip_dir in "${SCRIPT_DIR}"/custom_opp/*; do
    if [ -d "$chip_dir" ]; then
        EXTRA_OPP_PATHS="${chip_dir}:${EXTRA_OPP_PATHS}"
    fi
done

# 注册 Ascend Custom OPP (CANN 将会通过这个路径找到我们编译出来的算子)
if [ -n "$EXTRA_OPP_PATHS" ]; then
    export ASCEND_CUSTOM_OPP_PATH="${EXTRA_OPP_PATHS}${ASCEND_CUSTOM_OPP_PATH}"
fi

# 注册 TensorFlow 插件动态库 (.so) 路径以便 Python/TF 在运行时加载
if [ -d "${SCRIPT_DIR}/python_libs" ]; then
    export LD_LIBRARY_PATH="${SCRIPT_DIR}/python_libs:${LD_LIBRARY_PATH}"
fi

echo "=================================================================="
echo "[SUCCESS] RecSDK (rec_cust_ops) NPU Environment Loaded!"
echo "------------------------------------------------------------------"
echo "Detected Chips : $(ls -d "${SCRIPT_DIR}"/custom_opp/* 2>/dev/null | awk -F/ '{print $NF}' | tr '\n' ' ')"
echo "OPP_PATH       : ${ASCEND_CUSTOM_OPP_PATH}"
echo "LD_LIBRARY     : ${LD_LIBRARY_PATH}"
echo "=================================================================="

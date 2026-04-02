#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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

set -e

# ==============================================================================
# 1. 初始化路径
# ==============================================================================
readonly THIS_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
readonly WORK_DIR="$(dirname "${THIS_SCRIPT}")"
readonly UTILS_SCRIPT="${WORK_DIR}/../../../scripts/op_builder_utils.sh"

# ==============================================================================
# 2. 加载通用库
# ==============================================================================

if [ ! -f "$UTILS_SCRIPT" ]; then
    echo "ERROR: Cannot find op_builder_utils.sh at $UTILS_SCRIPT" >&2
    echo "Please check your directory structure." >&2
    exit 1
fi

source "$UTILS_SCRIPT"

# ==============================================================================
# 3. 参数配置
# ==============================================================================
vendor_name="hstu_dense_forward_fuxi"

parse_arguments "$@" || exit 1


echo "=========================================="
echo "Start Building Operator: ${vendor_name}"
echo "Target AI Core: ${ai_core}"
echo "Work Directory : ${WORK_DIR}"
echo "Source Root    : ${OP_ROOT_DIR}"
echo "=========================================="

# ==============================================================================
# 4. 执行标准化流程
# ==============================================================================

replace_operator_sources() {
# 清理旧文件 (防止残留)
    rm -rf "${WORK_DIR}/${vendor_name}/op_kernel"/*.h "${WORK_DIR}/${vendor_name}/op_kernel"/*.cpp 2>/dev/null || true
    rm -rf "${WORK_DIR}/${vendor_name}/op_host"/*.h "${WORK_DIR}/${vendor_name}/op_host"/*.cpp 2>/dev/null || true

    cp -rf "${WORK_DIR}/op_kernel" "${WORK_DIR}/${vendor_name}/"
    cp -rf "${WORK_DIR}/op_host"/*.h "${WORK_DIR}/${vendor_name}/op_host/"
    cp -rf "${WORK_DIR}/op_host"/hstu_*.cpp "${WORK_DIR}/${vendor_name}/op_host/"
    cp -rf "${WORK_DIR}/op_host"/tiling_policy.cpp "${WORK_DIR}/${vendor_name}/op_host/"
    cp -rf "${WORK_DIR}/op_host"/tiling_policy_factory.cpp "${WORK_DIR}/${vendor_name}/op_host/"
    if [ "$ai_core" = "ai_core-Ascend310P3" ]; then
    cp -rf "${WORK_DIR}/op_host"/tiling_policy_normal_v200_fuxi.cpp "${WORK_DIR}/${vendor_name}/op_host/"
      sed -i "1i #define SUPPORT_V200" ${WORK_DIR}/${vendor_name}/op_host/hstu_dense_forward_fuxi_tiling.h
      sed -i "1i #define SUPPORT_V200" ${WORK_DIR}/${vendor_name}/op_host/tiling_policy_define.h
      sed -i "1i #define SUPPORT_V200" ${WORK_DIR}/${vendor_name}/op_kernel/hstu_dense_forward_fuxi.cpp
    else
    cp -rf "${WORK_DIR}/op_host"/tiling_policy_jagged.cpp "${WORK_DIR}/${vendor_name}/op_host/"
    fi
}

with_onnx="true"

build_and_install_operator "$WORK_DIR" "$vendor_name" "$with_onnx" || exit 1
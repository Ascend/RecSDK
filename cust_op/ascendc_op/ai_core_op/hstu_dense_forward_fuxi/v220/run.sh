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
readonly THIS_DIR="$(dirname "${THIS_SCRIPT}")"
readonly WORK_DIR="${THIS_DIR}"
readonly OP_ROOT_DIR="$(dirname "${THIS_DIR}")"
readonly SRC_KERNEL_DIR="${WORK_DIR}/op_kernel"
readonly SRC_HOST_DIR="${WORK_DIR}/op_host"
readonly PROJECT_ROOT="$(dirname "$(dirname "$(dirname "${THIS_DIR}")")")"
readonly UTILS_SCRIPT="${PROJECT_ROOT}/scripts/op_builder_utils.sh"
readonly ONNX_PATH="${PROJECT_ROOT}/build/scripts/onnx_plugin"
readonly JSON_FILE="${ONNX_PATH}/json.hpp"

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

# 验证 AI Core
VALID_AI_CORES=(
    "ai_core-Ascend910B1"
    "ai_core-Ascend910B2"
    "ai_core-Ascend910B3"
    "ai_core-Ascend910B4"
    "ai_core-Ascend910_93"
    "ai_core-Ascend310P3"
)

validate_ai_core "$ai_core" || exit 1

# 检查系统环境和 CANN 版本
check_system_and_cann "$ai_core" || exit 1

# 生成算子代码
rm -rf "${WORK_DIR}/${vendor_name}"
msopgen gen -i ${vendor_name}.json -f tf -c ${ai_core} -lan cpp -out ./${vendor_name} -m 0 -op HstuDenseForwardFuxi

# 兼容cann9.0.0早期版本的老工程
if [ -d "${WORK_DIR}/${vendor_name}/cmake" ] && [ "${MAJOR_VERSION}" -eq 9 ]; then
    MAJOR_VERSION=8
fi

if [ "${MAJOR_VERSION}" -ge 9 ]; then
    overwrite_source_with_target "${WORK_DIR}/${vendor_name}" "${PROJECT_ROOT}/ai_core_op/custom_op_template" || exit 1
fi

# 定义生成后的目标目录
readonly TARGET_DIR="${WORK_DIR}/${vendor_name}"

# 【特异性逻辑】复制算子特有源码
echo "Copying specific operator source files to ${TARGET_DIR}..."

# 清理旧文件 (防止残留)
rm -rf "${TARGET_DIR}/op_kernel"/*.h "${TARGET_DIR}/op_kernel"/*.cpp 2>/dev/null || true
rm -rf "${TARGET_DIR}/op_host"/*.h "${TARGET_DIR}/op_host"/*.cpp 2>/dev/null || true

cp -rf op_kernel "${TARGET_DIR}/"
cp -rf op_host/*.h "${TARGET_DIR}/op_host/"
cp -rf op_host/hstu_*.cpp "${TARGET_DIR}/op_host/"
cp -rf op_host/tiling_policy.cpp "${TARGET_DIR}/op_host/"
cp -rf op_host/tiling_policy_factory.cpp "${TARGET_DIR}/op_host/"
if [ "$ai_core" = "ai_core-Ascend310P3" ]; then
  cp -rf op_host/tiling_policy_normal_v200_fuxi.cpp "${TARGET_DIR}/op_host/"
else
  cp -rf op_host/tiling_policy_jagged.cpp "${TARGET_DIR}/op_host/"
fi

# 构建 ONNX 适配层
build_onnx_adapter "$ai_core" "$ONNX_PATH" "$JSON_FILE" "$vendor_name" "$WORK_DIR" || exit 1

# 修改 CMakePresets.json
configure_cmake_presets "$vendor_name" "$ai_core" "$MAJOR_VERSION" "$TARGET_DIR" || exit 1

# CANN < 9.0 特殊处理并执行编译
prepare_legacy_build "$MAJOR_VERSION" "$vendor_name" "$TARGET_DIR" || exit 1

# 安装算子包
install_operator_package "$OS_ID" "$ARCH" "$TARGET_DIR" || exit 1

echo "=========================================="
echo "✅ Build & Install Successful for [${vendor_name}]!"
echo "=========================================="
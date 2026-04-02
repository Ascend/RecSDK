#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
source /etc/profile

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
    echo "ERROR: Cannot find op_builder_utils.sh at ${UTILS_SCRIPT}" >&2
    echo "Please check your directory structure." >&2
    exit 1
fi

source "$UTILS_SCRIPT"

# ==============================================================================
# 3. 参数配置（目录 fused_lazy_adam；CMake customize=mxrec_*；msopgen -op 由 MSOPGEN_OP_NAME 指定，避免默认 FusedLazyAdam）
# ==============================================================================
vendor_name="fused_lazy_adam"
export CMAKE_PRESET_VENDOR_NAME="mxrec_fused_lazy_adam"
export MSOPGEN_OP_NAME="LazyAdam"
export AI_CORE_PROFILE="v220"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/lazy_adam.json")"
export OPERATOR_SOURCE_ROOT="$(readlink -f "${WORK_DIR}")"
cp -f "$OPERATOR_JSON_FILE" "${OPERATOR_SOURCE_ROOT}/${CMAKE_PRESET_VENDOR_NAME}.json"

parse_arguments "$@" || exit 1

# ==============================================================================
# 4. 流程（-op LazyAdam；c310 下 configure_cmake_presets 会将 OPERATOR_JSON_FILE 拷至 CPack 路径）
# ==============================================================================
build_and_install_operator "$WORK_DIR" "$vendor_name" || exit 1
rm -rf "${OPERATOR_SOURCE_ROOT}/${CMAKE_PRESET_VENDOR_NAME}.json"
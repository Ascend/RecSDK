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
    echo "ERROR: Cannot find op_builder_utils.sh at ${UTILS_SCRIPT}" >&2
    echo "Please check your directory structure." >&2
    exit 1
fi

source "$UTILS_SCRIPT"

# ==============================================================================
# 3. 参数配置（与 v220 一致：enable_catlass 由环境传入 configure/prepare；另设 c310 专用路径）
# ==============================================================================
vendor_name="in_linear_silu"
export CATLASS_HOME="${WORK_DIR}/../../../../../third_party/catlass"
export AI_CORE_PROFILE="c310"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/../v220/in_linear_silu.json")"
export OPERATOR_SOURCE_ROOT="$(readlink -f "${WORK_DIR}/../v220")"
export INSERT_SUPPORT_950_PATHS="op_host/in_linear_silu.cpp"
catlass_include_dir=${CATLASS_HOME}/include
enable_catlass="True"

parse_arguments "$@" || exit 1

# ==============================================================================
# 4. 流程（msopgen -op 默认 PascalCase：InLinearSilu）
# ==============================================================================
build_and_install_operator "$WORK_DIR" "$vendor_name" || exit 1

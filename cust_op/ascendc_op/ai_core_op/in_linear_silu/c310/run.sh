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
# Catlass 依赖校验（与 v220/in_linear_silu/run.sh 一致）
# ==============================================================================
if [[ "x${CATLASS_HOME}x" == "xx" ]]; then
    if [[ -n "${CMAKE_SOURCE_DIR:-}" && -d "${CMAKE_SOURCE_DIR}/third_party/catlass" ]]; then
        CATLASS_HOME="${CMAKE_SOURCE_DIR}/third_party/catlass"
    elif [[ -n "${CMAKE_SOURCE_DIR:-}" && -d "${CMAKE_SOURCE_DIR}/catlass" ]]; then
        CATLASS_HOME="${CMAKE_SOURCE_DIR}/catlass"
    fi
fi
if [[ "x${CATLASS_HOME}x" == "xx" ]]; then
    echo "[ERROR] CATLASS_HOME not specified" >&2
    echo "Pls download CATLASS_HOME v1.3.0 from https://raw.gitcode.com/cann/catlass/archive/refs/heads/v1.3.0.zip" >&2
    exit 1
fi
if [[ ! -d "${CATLASS_HOME}" ]]; then
    echo "[ERROR] CATLASS_HOME directory does not exist: ${CATLASS_HOME}" >&2
    exit 1
fi

catlass_include_dir=${CATLASS_HOME}/include
enable_catlass="True"

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
export AI_CORE_PROFILE="c310"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/../v220/in_linear_silu.json")"
export OPERATOR_SOURCE_ROOT="$(readlink -f "${WORK_DIR}/../v220")"
export INSERT_SUPPORT_950_PATHS="op_host/in_linear_silu.cpp"

parse_arguments "$@" || exit 1

# ==============================================================================
# 4. 流程（msopgen -op 默认 PascalCase：InLinearSilu）
# ==============================================================================
build_and_install_operator "$WORK_DIR" "$vendor_name" || exit 1

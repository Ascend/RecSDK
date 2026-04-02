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
# 3. 参数配置
# ==============================================================================
vendor_name="hstu_dense_backward"
export AI_CORE_PROFILE="c310"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/../v220/hstu_dense_backward.json")"
export OPERATOR_SOURCE_ROOT="$(readlink -f "${WORK_DIR}/../v220")"

parse_arguments "$@" || exit 1

validate_ai_core "$ai_core" || exit 1
check_system_and_cann "$ai_core" || exit 1
gen_build_dir "$WORK_DIR" "$vendor_name" || exit 1

readonly TGT="${WORK_DIR}/${vendor_name}"
replace_operator_sources "${OPERATOR_SOURCE_ROOT}" "$TGT" || exit 1

sed -i "1i #define SUPPORT_950" "${TGT}/op_host/hstu_dense_backward.cpp"
sed -i "1i #define SUPPORT_950" "${TGT}/op_host/hstu_jagged_backward.cpp"
sed -i "1i #define __DAV_C310_VEC__" "${TGT}/op_host/hstu_dense_backward_tiling_common.h"
apply_op_kernel_compile_options_dual "$TGT" "-DENABLE_CV_COMM_VIA_SSBUF=true" || exit 1

configure_cmake_presets "$vendor_name" "$ai_core" "$MAJOR_VERSION" "$TGT" "False" || exit 1
prepare_and_build "$MAJOR_VERSION" "$vendor_name" "$TGT" "False" || exit 1
install_operator_package "$OS_ID" "$ARCH" "$TGT" || exit 1

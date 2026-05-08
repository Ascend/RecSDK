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
# 3. 参数配置（选择性拷贝 v220 源码，见 replace_sources.sh）
# ==============================================================================
vendor_name="hstu_dense_forward_fuxi"
export AI_CORE_PROFILE="c310"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/../v220/hstu_dense_forward_fuxi.json")"

parse_arguments "$@" || exit 1

# ==============================================================================
# 4. 执行标准化流程
# ==============================================================================
validate_ai_core "$ai_core" || exit 1
check_system_and_cann "$ai_core" || exit 1
gen_build_dir "$WORK_DIR" "$vendor_name" || exit 1

readonly TGT="${WORK_DIR}/${vendor_name}"
readonly V220="$(readlink -f "${WORK_DIR}/../v220")"

rm -rf "${TGT}/op_kernel"/*.h "${TGT}/op_kernel"/*.cpp 2>/dev/null || true
rm -rf "${TGT}/op_host"/*.h "${TGT}/op_host"/*.cpp 2>/dev/null || true

cp -rf "${V220}/op_kernel/"* "${TGT}/op_kernel/"
cp -rf "${V220}/op_host"/*.h "${TGT}/op_host/"
cp -rf "${V220}/op_host"/hstu_*.cpp "${TGT}/op_host/"
cp -rf "${V220}/op_host/tiling_policy.cpp" "${TGT}/op_host/"
cp -rf "${V220}/op_host/tiling_policy_factory.cpp" "${TGT}/op_host/"
cp -rf "${V220}/op_host/tiling_policy_jagged.cpp" "${TGT}/op_host/"

sed -i "1i #define SUPPORT_950" "${TGT}/op_host/hstu_dense_forward_fuxi.cpp"
sed -i "1i #define __DAV_C310_VEC__" "${TGT}/op_host/tiling_policy_define.h"

configure_cmake_presets "$vendor_name" "$ai_core" "$BUILD_VERSION" "$TGT" "False" || exit 1
prepare_and_build "$BUILD_VERSION" "$vendor_name" "$TGT" "False" || exit 1
install_operator_package "$OS_ID" "$ARCH" "$TGT" || exit 1

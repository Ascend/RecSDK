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
# 3. 参数配置（c310/op_kernel + v220 部分文件混合，见 replace_sources.sh）
# ==============================================================================
vendor_name="hstu_dense_forward"
export AI_CORE_PROFILE="c310"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/../v220/hstu_dense_forward.json")"

parse_arguments "$@" || exit 1

# ==============================================================================
# 4. 执行标准化流程
# ==============================================================================
validate_ai_core "$ai_core" || exit 1
check_system_and_cann "$ai_core" || exit 1
gen_build_dir "$WORK_DIR" "$vendor_name" || exit 1

readonly TGT="${WORK_DIR}/${vendor_name}"
readonly V220="$(readlink -f "${WORK_DIR}/../v220")"
readonly C310K="$(readlink -f "${WORK_DIR}/op_kernel")"

rm -rf "${TGT}/op_kernel"/*.h "${TGT}/op_kernel"/*.cpp 2>/dev/null || true
rm -rf "${TGT}/op_host"/*.h "${TGT}/op_host"/*.cpp 2>/dev/null || true

cp -rf "${V220}/op_host"/*.h "${TGT}/op_host/"
cp -rf "${V220}/op_host"/hstu_*.cpp "${TGT}/op_host/"
cp -rf "${V220}/op_host/tiling_policy.cpp" "${TGT}/op_host/"
cp -rf "${V220}/op_host/tiling_policy_dense.cpp" "${TGT}/op_host/"
cp -rf "${V220}/op_host/tiling_policy_jagged.cpp" "${TGT}/op_host/"
cp -rf "${V220}/op_host/tiling_policy_paged.cpp" "${TGT}/op_host/"

cp -rf "${C310K}/"* "${TGT}/op_kernel/"
cp -rf "${V220}/op_kernel/hstu_common_const.h" "${TGT}/op_kernel/"
cp -rf "${V220}/op_kernel/hstu_dense_causal_mask.h" "${TGT}/op_kernel/"
cp -rf "${V220}/op_kernel/hstu_split_core_policy.h" "${TGT}/op_kernel/"

for _f in \
    op_kernel/hstu_common_const.h \
    op_kernel/matmul_constexpr.h \
    op_kernel/hstu_dense_forward_jagged_kernel.h \
    op_kernel/hstu_paged_forward_kernel.h \
    op_host/hstu_jagged_forward.cpp \
    op_host/hstu_paged_forward.cpp \
    op_host/hstu_dense_forward.cpp \
    op_host/tiling_policy.cpp; do
    sed -i "1i #define SUPPORT_950" "${TGT}/${_f}"
done

add_line='install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../../v220/hstu_dense_forward.json DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic)'
sed -i "\$a\\${add_line}" "${TGT}/op_kernel/CMakeLists.txt"
apply_op_kernel_compile_options_dual "$TGT" "-DENABLE_CV_COMM_VIA_SSBUF=true" || exit 1

configure_cmake_presets "$vendor_name" "$ai_core" "$MAJOR_VERSION" "$TGT" "False" || exit 1
prepare_and_build "$MAJOR_VERSION" "$vendor_name" "$TGT" "False" || exit 1
install_operator_package "$OS_ID" "$ARCH" "$TGT" || exit 1

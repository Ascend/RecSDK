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
# 3. 参数配置（目录 fused_sgd；CMake customize=mxrec_fused_agd；MSOPGEN_OP_NAME=Sgd 避免默认 FusedSgd）
# ==============================================================================
vendor_name="fused_sgd"
export with_onnx="true"
export CMAKE_PRESET_VENDOR_NAME="mxrec_sgd"
export MSOPGEN_OP_NAME="Sgd"
export AI_CORE_PROFILE="c310"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/../v220/sgd.json")"
export OPERATOR_SOURCE_ROOT="$(readlink -f "${WORK_DIR}/../v220/")"
readonly TGT="${WORK_DIR}/${vendor_name}"
cp -f "${OPERATOR_JSON_FILE}" "${OPERATOR_SOURCE_ROOT}/${CMAKE_PRESET_VENDOR_NAME}.json"

parse_arguments "$@" || exit 1

# ==============================================================================
# 4. 流程（-op Sgd）
# ==============================================================================
validate_ai_core "$ai_core" || exit 1
check_system_and_cann "$ai_core" || exit 1

rm -rf "${TGT}"
msopgen gen -i "${OPERATOR_JSON_FILE}" -f tf -c "${ai_core}" -lan cpp -out "${TGT}" -m 0 -op "${MSOPGEN_OP_NAME}"

set_build_version "${TGT}"

replace_operator_sources "${OPERATOR_SOURCE_ROOT}" "${TGT}" || exit 1
sed -i "1i #define SUPPORT_950" "${TGT}/op_host/sgd.cpp"
cp -f "${OPERATOR_SOURCE_ROOT}"/../../common/* "${TGT}"/op_host/
configure_cmake_presets "${CMAKE_PRESET_VENDOR_NAME}" "${ai_core}" "${BUILD_VERSION}" "${TGT}" || exit 1
prepare_and_build "$BUILD_VERSION" "${CMAKE_PRESET_VENDOR_NAME}" "${TGT}" || exit 1
install_operator_package "$OS_ID" "$ARCH" "${TGT}" || exit 1
rm -rf "${WORK_DIR}/${CMAKE_PRESET_VENDOR_NAME}.json"

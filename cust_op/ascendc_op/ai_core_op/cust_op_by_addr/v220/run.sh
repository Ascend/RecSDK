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
# 3. 参数配置
# ==============================================================================
vendor_name="cust_op_by_addr"
export with_onnx="true"
export AI_CORE_PROFILE="v220"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/emb_custom.json")"
readonly V220_SRC="$(readlink -f "${WORK_DIR}")"
readonly TGT="${WORK_DIR}/${vendor_name}"

parse_arguments "$@" || exit 1

cp -f "$OPERATOR_JSON_FILE" "${V220_SRC}/${vendor_name}.json"

# ==============================================================================
# 4. 执行标准化流程（不可整段走 build_and_install_operator：需连续两次 msopgen -m 0 / -m 1）
# ==============================================================================
validate_ai_core "$ai_core" || exit 1
check_system_and_cann "$ai_core" || exit 1

rm -rf "${TGT}"
msopgen gen -i "${OPERATOR_JSON_FILE}" -f tf -c "${ai_core}" -lan cpp -out "${TGT}" -m 0 -op EmbeddingLookupByAddress || exit 1
msopgen gen -i "${OPERATOR_JSON_FILE}" -f tf -c "${ai_core}" -lan cpp -out "${TGT}" -m 1 -op EmbeddingUpdateByAddress || exit 1

if [ -d "${TGT}/cmake" ] && [ "${MAJOR_VERSION}" -eq 9 ]; then
    export MAJOR_VERSION=8
fi

replace_operator_sources "${V220_SRC}" "${TGT}" || exit 1

configure_cmake_presets "$vendor_name" "$ai_core" "$MAJOR_VERSION" "${TGT}" || exit 1
prepare_and_build "$MAJOR_VERSION" "$vendor_name" "${TGT}" || exit 1
install_operator_package "$OS_ID" "$ARCH" "${TGT}" || exit 1

rm -f "${V220_SRC}/${vendor_name}.json"

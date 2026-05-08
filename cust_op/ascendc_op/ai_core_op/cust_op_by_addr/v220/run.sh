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
export ENABLE_SOURCE_PACKAGE="true"
readonly V220_SRC="$(readlink -f "${WORK_DIR}")"

parse_arguments "$@" || exit 1

cp -f "$OPERATOR_JSON_FILE" "${V220_SRC}/${vendor_name}.json"

# ==============================================================================
# 4. 执行标准化流程
# ==============================================================================

gen_build_dir() {
    local work_dir="$1"
    local vendor_name="$2"

    rm -rf "${work_dir}/${vendor_name}"
    local json_file="${OPERATOR_JSON_FILE:-${work_dir}/${vendor_name}.json}"
    msopgen gen -i "${json_file}" -f tf -c "${ai_core}" -lan cpp -out "${work_dir}/${vendor_name}" -m 0 \
        -op EmbeddingLookupByAddress || exit 1
    msopgen gen -i "${json_file}" -f tf -c "${ai_core}" -lan cpp -out "${work_dir}/${vendor_name}" -m 1 \
        -op EmbeddingUpdateByAddress || exit 1

    set_build_version "${work_dir}/${vendor_name}"

    if [ "${BUILD_VERSION}" = "modern" ]; then
        overwrite_source_with_target "${work_dir}/${vendor_name}" \
        "${__PROJECT_ROOT}/ai_core_op/custom_op_template" || return 1
    fi
}

build_and_install_operator "$WORK_DIR" "$vendor_name" || exit 1
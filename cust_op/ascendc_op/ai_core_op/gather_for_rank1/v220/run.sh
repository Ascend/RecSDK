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
vendor_name="gather_for_rank1"

parse_arguments "$@" || exit 1

# ==============================================================================
# 4. 执行标准化流程
# ==============================================================================
with_onnx="true"

replace_operator_sources() {
    local src_dir="$1"
    local tgt_dir="$2"

    # 清理旧文件 (防止残留)
    rm -rf "${tgt_dir}/op_kernel"/*.h "${tgt_dir}/op_kernel"/*.cpp 2>/dev/null || true
    rm -rf "${tgt_dir}/op_host"/*.h "${tgt_dir}/op_host"/*.cpp 2>/dev/null || true

    # 复制新文件
    cp -rf ${src_dir}/op_kernel/* "${tgt_dir}/op_kernel/"
    cp -rf ${src_dir}/op_host/* "${tgt_dir}/op_host/"
    if [ "$ai_core" = "ai_core-Ascend310P3" ] && [ "$MAJOR_VERSION" -lt 9 ]; then
        sed -i "1i #define SUPPORT_V200" "${tgt_dir}/op_host/gather_for_rank1.cpp"
        sed -i "1i #define SUPPORT_V200" "${tgt_dir}/op_kernel/gather_for_rank1_kernel.h"
    fi
    return 0
}

build_and_install_operator "$WORK_DIR" "$vendor_name" "$with_onnx" || exit 1

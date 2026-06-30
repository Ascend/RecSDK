#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

# ============================================================
# HSTU Dense Forward Test — 一键构建 & 运行
#   Usage: ./build.sh [num_cards] [batch_size] [seq_len] [skip_golden]
#   skip_golden: skip|1 = 跳过 golden 数据生成
#   1. 生成 golden 数据 (torchrun N进程)
#   2. 编译 C++ 测试 (cmake + make)
#   3. 运行 N 卡测试
# ============================================================

NUM_CARDS="${1:-8}"
BATCH_SIZE="${2:-31}"
SEQ_LEN="${3:-1024}"
SKIP_GOLDEN="${4:-0}"
echo "Testing with ${NUM_CARDS} card(s), bs=${BATCH_SIZE}, seq=${SEQ_LEN}, skip_golden=${SKIP_GOLDEN}..."

# 环境检查
if [ -z "${ASCEND_TOOLKIT_HOME}" ]; then
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    elif [ -f /usr/local/Ascend/latest/bin/setenv.bash ]; then
        source /usr/local/Ascend/latest/bin/setenv.bash
    else
        echo "ERROR: ASCEND_TOOLKIT_HOME not set." >&2
        exit 1
    fi
fi

CANN="${ASCEND_TOOLKIT_HOME}"
OPV="${CANN}/opp/vendors/hstu_dense_forward"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- Step 1: 生成 Golden 数据 ----------
if [ "${SKIP_GOLDEN}" = "skip" ] || [ "${SKIP_GOLDEN}" = "1" ]; then
    echo "=========================================="
    echo "Step 1: Skipping golden data generation."
    echo "=========================================="
else
    echo "=========================================="
    echo "Step 1: Generating golden data (${NUM_CARDS} proc)..."
    echo "=========================================="
    rm -rf bin_file
    mkdir -p bin_file
    source "${CANN}/bin/setenv.bash"

    # 检查 torchrun 是否在 PATH 中（尝试自动激活 conda 环境）
    if ! command -v torchrun &>/dev/null; then
        CONDA_BASE=""
        if command -v conda &>/dev/null; then
            CONDA_BASE="$(conda info --base 2>/dev/null)" || true
        fi
        if [ -z "${CONDA_BASE}" ] && [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="${HOME}/miniconda3"
        elif [ -z "${CONDA_BASE}" ] && [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="/opt/conda"
        fi
        if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
            source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
            if conda env list 2>/dev/null | grep -q "torch_npu"; then
                conda activate torch_npu 2>/dev/null || true
            fi
        fi
        if ! command -v torchrun &>/dev/null; then
            echo "ERROR: torchrun not found in PATH." >&2
            echo "Please activate your pytorch environment first, e.g.:" >&2
            echo "  conda activate <env_name>" >&2
            exit 1
        fi
    fi

    torchrun --nproc_per_node=${NUM_CARDS} generate_golden.py --bs ${BATCH_SIZE} --seq ${SEQ_LEN}
    echo "Golden data generated."
fi

# ---------- Step 2: 编译 ----------
echo ""
echo "=========================================="
echo "Step 2: Building test..."
echo "=========================================="
cmake . || { echo "ERROR: cmake failed" >&2; exit 1; }
make || { echo "ERROR: make failed" >&2; exit 1; }
echo ""
echo "Build success: $(pwd)/test_hstu_dense_forward"
cd "${SCRIPT_DIR}"

# ---------- Step 3: 运行测试 ----------
echo ""
echo "=========================================="
echo "Step 3: Running ${NUM_CARDS}-card test..."
echo "=========================================="
export LD_LIBRARY_PATH="${OPV}/op_api/lib/:${LD_LIBRARY_PATH}"
./test_hstu_dense_forward ${NUM_CARDS} ${SCRIPT_DIR}/bin_file ${BATCH_SIZE} ${SEQ_LEN}
echo ""
echo "=========================================="
echo "All done."
echo "=========================================="

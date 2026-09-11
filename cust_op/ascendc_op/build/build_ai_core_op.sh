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
#
# Perf: 把算子编译从「逐个串行 for 循环」改为「后台任务并行」，默认 4 路并发。
# 其余逻辑（算子列表、A2/A3/A5/310P/A2-TF 分支、REBUILD_ALL、ERROR_MODE、产物打包）
# 与原版保持一致。ccache 逻辑不变（若机器上预装了 ccache 则自动启用，否则走默认编译器）。
#
# 用法（与原版兼容）：
#   bash build_ai_core_op.sh A2                          # 默认 4 路并行
#   bash build_ai_core_op.sh A2 true continue 8          # 显式指定 8 路并行
#
# 环境变量：
#   PARALLEL_LEVEL   并发算子数（覆盖命令行第 4 参数；默认 4，上限受真实 CPU 核数钳制）
#

set -e

function setup_ccache()
{
    # 检查 ccache 是否安装
    if ! command -v ccache &> /dev/null; then
        echo "[INFO] ccache not found, using default compiler..."
        return 0
    fi

    echo "[INFO] ccache found, enabling compiler cache..."

    # 设置 ccache 路径到 PATH 最前面
    if [ -d "/usr/lib/ccache" ]; then
        export PATH=/usr/lib/ccache:$PATH
    elif [ -d "/usr/lib64/ccache" ]; then
        export PATH=/usr/lib64/ccache:$PATH
    else
        echo "Warning: ccache directory not found in standard paths."
    fi

    # 设置缓存目录和大小
    export CCACHE_DIR="${CCACHE_DIR:-/home/cache}"
    export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-10G}"
    export CCACHE_COMPRESS=true
    export CCACHE_HASHDIR=true
    # 将所有绝对路径归一化到项目根目录，跨编译目录才能命中
    export CCACHE_BASEDIR=$(pwd)
    export CCACHE_SLOPPINESS="time_macros,file_macro,include_file_mtime,include_file_ctime,pch_defines,locale,random_seed"
    export CCACHE_COMPILERCHECK=content
    export CC_COMPILER_LAUNCHER=ccache
    export CXX_COMPILER_LAUNCHER=ccache
}

if [ "$#" -lt 1 ]; then
    echo "ERROR: Please specify the version to compile. e.g. 'bash $0 A2'"
    exit 1
fi

BUILD_VER=${1}

if [[ "${BUILD_VER}" =~ ^(A2|A3|A5|310P|A2-TF)$ ]]; then
    echo "BUILD_VER: ${BUILD_VER}"
else
    echo "ERROR: Unknown BUILD_VER:${BUILD_VER}"
    exit 1
fi

REBUILD_ALL=${2:-"true"}

if [[ "${REBUILD_ALL}" == "true" ]]; then
    echo "Rebuild all operators: ${REBUILD_ALL}, \
        you can set it to false to only build operators that have not been built successfully. \
        e.g. 'bash $0 A2 false'"
elif [[ "${REBUILD_ALL}" == "false" ]]; then
    echo "Only build operators that have not been built successfully: rebuild_all=${REBUILD_ALL}"
else
    echo "ERROR: Unknown value for REBUILD_ALL: ${REBUILD_ALL}, please set it to true or false."
    exit 1
fi

ERROR_MODE=${3:-"exit"}
if [[ "${ERROR_MODE}" == "exit" ]]; then
    echo "Error mode: ${ERROR_MODE}, the script will exit immediately when an error occurs during compilation.
          you can set it to continue to record the failed operator and continue to compile the remaining operators. \
          e.g. 'bash $0 A2 false continue'"
elif [[ "${ERROR_MODE}" == "continue" ]]; then
    echo "Error mode: ${ERROR_MODE}, the script will record the failed operator and continue to compile the remaining operators when an error occurs."
else
    echo "ERROR: Unknown value for ERROR_MODE: ${ERROR_MODE}, please set it to exit or continue."
    exit 1
fi

# ---------------------------------------------------------------------------
# 探测「真实」可用 CPU 核数。容器/流水线里 `nproc` 常返回宿主机核数（本机实测
# nproc=128），而实际配额由 cgroup 限制（实测 CPU limit=50 cores、内存 90GiB），
# 只有按 cgroup 配额分核才有意义。优先 cgroup v2/v1，回退 nproc。
# ---------------------------------------------------------------------------
function _detect_real_cpu() {
    local quota period cores=""
    # cgroup v2: /sys/fs/cgroup/cpu.max => "quota period"（quota 可为 "max"）
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null || true
        if [ -n "${quota}" ] && [ "${quota}" != "max" ] && [ -n "${period}" ] && [ "${period}" -gt 0 ]; then
            cores=$(( (quota + period - 1) / period ))
        fi
    fi
    # cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us
    if [ -z "${cores}" ] || [ "${cores}" -le 0 ]; then
        if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
            quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo 0)
            period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 0)
            if [ -n "${quota}" ] && [ "${quota}" -gt 0 ] && [ -n "${period}" ] && [ "${period}" -gt 0 ]; then
                cores=$(( (quota + period - 1) / period ))
            fi
        fi
    fi
    if [ -z "${cores}" ] || [ "${cores}" -le 0 ]; then
        cores=$(nproc 2>/dev/null || echo 1)
        # cgroup 未给出有效有限配额（cpu.max/cfs_quota 为 max 或文件不存在的场景），
        # 此时回退宿主机 nproc 会拿到共享节点全量核数（如 128），极易超卖导致 OOM。
        # 对回退值保守封顶（可用 RECSDK_MAX_FALLBACK_CPU 环境变量覆盖）。
        local cap="${RECSDK_MAX_FALLBACK_CPU:-16}"
        [ "${cores}" -gt "${cap}" ] && cores="${cap}"
    fi
    echo "${cores}"
}

# ---------------------------------------------------------------------------
# 探测「真实」可用内存（GiB）。与 CPU 同理：容器里 free 常返回宿主内存而非 cgroup
# 配额，只有 cgroup 配额才算数。优先 v2 memory.max / v1 limit_in_bytes，回退 free。
# ---------------------------------------------------------------------------
function _detect_real_mem_gb() {
    local limit="" mem_gb=0
    if [ -r /sys/fs/cgroup/memory.max ]; then
        limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)
    fi
    if [ -z "${limit}" ] || [ "${limit}" = "max" ]; then
        if [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
            limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)
        fi
    fi
    case "${limit}" in
        ""|"max") limit=$(free -b 2>/dev/null | awk '/^Mem:/{print $2}') ;;
    esac
    if [ -n "${limit}" ] && [ "${limit}" -gt 0 ] 2>/dev/null; then
        mem_gb=$(( limit / 1024 / 1024 / 1024 ))
    fi
    [ "${mem_gb}" -le 0 ] && mem_gb=8
    echo "${mem_gb}"
}

# ---------------------------------------------------------------------------
# 并行度（优先级：CLI 第 4 参数 > PARALLEL_LEVEL 环境变量 > 按真实资源自动推导）。
# 总原则：并发相关设置一律基于「实际机器」cgroup 配额动态推导，绝不写死核数/内存/并行数。
# ---------------------------------------------------------------------------
NCPU=$(_detect_real_cpu)
NMEM_GB=$(_detect_real_mem_gb)
echo "Usable resources (cgroup-aware): ${NCPU} cores, ${NMEM_GB} GiB RAM"

PARALLEL_LEVEL_CLI=${4:-}
if [ -n "${PARALLEL_LEVEL_CLI}" ]; then
    PARALLEL_LEVEL="${PARALLEL_LEVEL_CLI}"
elif [ -n "${PARALLEL_LEVEL:-}" ]; then
    PARALLEL_LEVEL="${PARALLEL_LEVEL}"
else
    # 自动并行度 = min( ceil(NCPU/每算子核数), 内存/每算子内存 )。
    #   - 每算子建议核数 / 内存预留量是「可覆盖的策略系数」（非机器量），默认偏保守，
    #     让 50 核/90GiB 大池子得到 4 路（既有验证甜点）、小池子自动降为串行兜底。
    OP_CORES_PER_OP=${OP_CORES_PER_OP:-16}
    OP_MEM_PER_OP_GB=${OP_MEM_PER_OP_GB:-16}
    [ "${OP_CORES_PER_OP}" -le 0 ] && OP_CORES_PER_OP=1
    [ "${OP_MEM_PER_OP_GB}" -le 0 ] && OP_MEM_PER_OP_GB=1
    MAX_BY_CPU=$(( (NCPU + OP_CORES_PER_OP - 1) / OP_CORES_PER_OP ))
    MAX_BY_MEM=$(( NMEM_GB / OP_MEM_PER_OP_GB ))
    PARALLEL_LEVEL="${MAX_BY_CPU}"
    [ "${MAX_BY_MEM}" -gt 0 ] && [ "${MAX_BY_MEM}" -lt "${PARALLEL_LEVEL}" ] && PARALLEL_LEVEL="${MAX_BY_MEM}"
    [ "${PARALLEL_LEVEL}" -lt 1 ] && PARALLEL_LEVEL=1
    echo "[INFO] auto PARALLEL_LEVEL=${PARALLEL_LEVEL} (by-cpu=ceil(${NCPU}/${OP_CORES_PER_OP})=${MAX_BY_CPU}, by-mem=${NMEM_GB}/${OP_MEM_PER_OP_GB}=${MAX_BY_MEM})"
fi

# 钳制到真实核数，避免超配
if [ "${PARALLEL_LEVEL}" -gt "${NCPU}" ]; then
    echo "[INFO] clamp PARALLEL_LEVEL ${PARALLEL_LEVEL} -> ${NCPU}"
    PARALLEL_LEVEL="${NCPU}"
fi
echo "Parallel level: ${PARALLEL_LEVEL} operator(s) in parallel"

# ---------------------------------------------------------------------------
# 并行构建配套开关：
#   1) RECSDK_OP_SKIP_SYS_INSTALL=1 —— 跳过把算子 .run 安装到系统 CANN 目录
#      （多算子并发安装会竞态；whl 打包只依赖 build_out 下的 .run，无需系统安装）。
#   2) RECSDK_OP_BUILD_JOBS —— 单个算子内部 cmake --build 的 -j 线程数。
#      不再按 NCPU/PARALLEL_LEVEL 均分：均分会饿死大算子——实测基线每算子
#      用满 -j 时 hstu_dense_backward 仅 87s，而 -j32 时增至 298s（近乎线性翻 4 倍）。
#      这里给每个算子最多「真实核数」的内层 -j，外层并发由 OS 时间片在真实核数上共享；
#      内存实测 90GiB，无 OOM 风险。若想更保守，可用 RECSDK_OP_BUILD_JOBS 环境变量覆盖。
# ---------------------------------------------------------------------------
export RECSDK_OP_SKIP_SYS_INSTALL=1
if [ -n "${NCPU:-}" ] && [ "${NCPU}" -gt 0 ]; then
    [ -z "${RECSDK_OP_BUILD_JOBS:-}" ] && RECSDK_OP_BUILD_JOBS="${NCPU}"
    export RECSDK_OP_BUILD_JOBS
    echo "Per-op build jobs: ${RECSDK_OP_BUILD_JOBS} (${PARALLEL_LEVEL} ops share ${NCPU} real cores via OS time-slicing)"
fi

ARCH="$(uname -m)"
CUR_DIR=$(dirname "$(readlink -f "$0")")
ASCENDC_OP_DIR=$(dirname "${CUR_DIR}")
torch_plugin_path="${ASCENDC_OP_DIR}"/../framework/torch_plugin
ops_path="${ASCENDC_OP_DIR}"/ai_core_op
base_op_dir="v220"

OPS_BUILD_LOG_DIR="${ASCENDC_OP_DIR}/output/build_logs"
mkdir -p "${OPS_BUILD_LOG_DIR}"
FAILED_LOG="${OPS_BUILD_LOG_DIR}/_failed_ops.txt"
TIMING_FILE="${OPS_BUILD_LOG_DIR}/_timing.tsv"
: > "${FAILED_LOG}"
: > "${TIMING_FILE}"

source /etc/profile

declare -A OP_PLUGIN_MAP=(
  ["asynchronous_complete_cumsum"]="asynchronous_complete_cumsum"
  ["recops_backward_codegen_adagrad_unweighted_exact"]="split_embedding_codegen_forward_unweighted"
  ["block_bucketize_sparse_features"]="block_bucketize_sparse_features"
  ["concat_jagged_tensor"]="concat_2d_jagged"
  ["concat_jagged_tensor_grad"]="concat_2d_jagged"
  ["dense_embedding_codegen_lookup_function"]="dense_embedding_codegen_lookup_function"
  ["dense_embedding_codegen_lookup_function_grad"]="dense_embedding_codegen_lookup_function"
  ["dense_to_jagged"]="dense_to_jagged"
  ["disentangle_attention"]="disentangle_attention"
  ["expand_into_jagged_permute"]="expand_into_jagged_permute"
  ["gather_for_rank1"]="gather_for_rank1"
  ["hstu_dense_backward"]="hstu"
  ["hstu_dense_backward_fuxi"]="hstu_dense_backward_fuxi"
  ["hstu_dense_forward"]="hstu"
  ["hstu_dense_forward_fuxi"]="hstu_dense_forward_fuxi"
  ["hstu_v2"]="hstu_v2"
  ["in_linear_silu"]="in_linear_silu"
  ["in_linear_silu_backward"]="in_linear_silu"
  ["index_select_for_rank1_backward"]="gather_for_rank1"
  ["int_nbit_split_embedding_codegen_lookup_function"]="int_nbit_split_embedding_codegen_lookup_function"
  ["invert_permute"]="invert_permute"
  ["jagged_to_padded_dense"]="jagged_to_padded_dense"
  ["ln_mul"]="ln_mul"
  ["multislice_concat"]="multislice_concat"
  ["norm_multiply_dropout"]="norm_multiply_dropout"
  ["offsets_range"]="offsets_range"
  ["permute_pooled_embs"]="permute_pooled_embs"
  ["permute2d_sparse_data"]="permute1d_sparse_data permute2d_sparse_data"
  ["relative_attn_bias_backward"]="relative_attn_bias"
  ["relative_attn_bias_pos"]="relative_attn_bias"
  ["relative_attn_bias_time"]="relative_attn_bias"
  ["reverse_sequence"]="reverse_sequence"
  ["segment_sum_csr"]="segment_sum_csr"
  ["recops_split_embedding_codegen_forward_unweighted"]="split_embedding_codegen_forward_unweighted"
  ["token_mixing"]="token_mixing"
  ["select_dim1_to_permute"]="keyed_jagged_index_select_dim1"
)


support_A2_tf_ops="cust_op_by_addr
fused_lazy_adam
fused_sgd
lccl
pcie_through
"
support_A3_list="asynchronous_complete_cumsum
gather_for_rank1
index_select_for_rank1_backward
dense_to_jagged
jagged_to_padded_dense
permute_pooled_embs
permute2d_sparse_data
recops_split_embedding_codegen_forward_unweighted
recops_backward_codegen_adagrad_unweighted_exact
hstu_dense_forward_fuxi
hstu_dense_backward_fuxi
disentangle_attention
dense_embedding_codegen_lookup_function
dense_embedding_codegen_lookup_function_grad
hstu_dense_forward
hstu_dense_backward
in_linear_silu
"
support_310p_list="gather_for_rank1
hstu_dense_forward_fuxi
relative_attn_bias_time
relative_attn_bias_pos
"

cd "${ASCENDC_OP_DIR}"

function cp_op_plugin()
{
    cd "${torch_plugin_path}"
    cp -r torch_library/common "${plugin_output_path}"
}

function make_output_dir() {
    mxrec_output_path="${ASCENDC_OP_DIR}"/output
    output_path="${CUR_DIR}"/output
    opp_output_path="${CUR_DIR}"/output/recsdk_ops
    plugin_output_path="${CUR_DIR}"/output/torch_plugin/torch_library
    mkdir -p "${mxrec_output_path}"
    mkdir -p "${output_path}"
    mkdir -p "${opp_output_path}"
    mkdir -p "${plugin_output_path}"
}

function in_list() {
    local w=$1; shift
    for x; do [ "$x" = "$w" ] && return 0; done
    return 1
}

function record_failed_op() {
    local op_name="$1"
    echo "Failed to build operator: ${op_name}"
    # 失败记录写入临时文件，跨后台任务安全（后台任务无法回写主进程数组）
    echo "${op_name}" >> "${FAILED_LOG}"
}

# ---------------------------------------------------------------------------
# 拷贝算子对应的 torch_plugin 适配层目录（线程/进程安全）。
# 多个算子可能映射到同一个 plugin 目录（如 hstu_dense_forward/backward 都是 hstu），
# 并行编译时并发 cp -r 同名目录会互相嵌套/损坏。用 mkdir 原子锁串行化同名目录，
# 且已有目标时跳过，保证幂等。
# ---------------------------------------------------------------------------
function _copy_plugin_dirs() {
    local plugin_list="$1"
    local plugin_dir_name _lock
    mkdir -p "${plugin_output_path}"
    for plugin_dir_name in ${plugin_list}; do
        [ -z "${plugin_dir_name}" ] && continue
        _lock="${plugin_output_path}/.lock_${plugin_dir_name}"
        while ! mkdir "${_lock}" 2>/dev/null; do
            sleep 0.1
        done
        if [ ! -e "${plugin_output_path}/${plugin_dir_name}" ]; then
            cp -r "${torch_plugin_path}/torch_library/${plugin_dir_name}" "${plugin_output_path}/${plugin_dir_name}"
        fi
        rmdir "${_lock}" 2>/dev/null || true
    done
}

# ---------------------------------------------------------------------------
# 收集待编译算子目录，输出 "dir_name:dir_abs_path" 列表到 _list_file
# 按「源码体积」降序排列：大算子（hstu 系列动辄 3~5 分钟）先编译，小算子随后
# 回填到 wait -n 释放出的空闲槽位，最大化并行重叠、缩短关键路径。
# 原版按目录 glob 顺序（≈字母序）会把 hstu 挤到中段，拉出 85s 头 + 152s 尾。
# ---------------------------------------------------------------------------
function _collect_ops() {
    local _arch="$1"
    local _basedir="$2"
    local _list_file="$3"
    local _tmp="${_list_file}.unsorted"
    : > "${_tmp}"
    local _dir _dname _size
    for _dir in "${_basedir}"/*; do
        [ -d "${_dir}" ] || continue
        _dname=$(basename "${_dir}")
        case "${_dname}" in
            cmake|common|custom_op_template) continue ;;
        esac
        if [ -d "${_dir}/${_arch}" ]; then
            _size=$(find "${_dir}/${_arch}" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
            printf '%s\t%s:%s\n' "${_size}" "${_dname}" "${_dir}" >> "${_tmp}"
        fi
    done
    # 体积降序输出 "dir_name:dir_abs_path"
    sort -rn "${_tmp}" | awk -F'\t' '{print $2}' > "${_list_file}"
    rm -f "${_tmp}"
}

# ---------------------------------------------------------------------------
# 单个算子编译（等价于原版 compile_ops_v220/compile_ops_A5 中对一个 dir 的处理）
# 输入 "dir_name:dir_abs_path"；用 subshell 隔离 cd
# ---------------------------------------------------------------------------
function _compile_one_op() {
    local item="$1"
    local dir_name="${item%%:*}"
    local dir="${item#*:}"
    local _subdir="${base_op_dir}"
    if [ "${BUILD_VER}" == "A5" ]; then
        _subdir="c310"
    fi
    local cur_ver_op_dir="${dir}/${_subdir}"
    local plugin_dir_names="${OP_PLUGIN_MAP[$dir_name]}"

    local _log="${OPS_BUILD_LOG_DIR}/${dir_name}.log"
    local _t0 _t1 _rc=0 _elapsed
    _t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] START ${dir_name}"

    (
        set +e
        cd "${cur_ver_op_dir}" || { echo "cd failed: ${cur_ver_op_dir}"; exit 3; }
        if [ "${BUILD_VER}" == "310P" ]; then
            for item in $support_310p_list; do
                if [ "$item" == "$dir_name" ]; then
                    if [ "${REBUILD_ALL}" == "false" ] && \
                        [ -f "${opp_output_path}"/mxrec_opp_"${dir_name}"_310p.run ]; then
                        echo "Operator ${dir_name} for 310P already built, skipping..."
                        continue
                    fi
                    bash ./run.sh --ai-core ai_core-Ascend310P3 || {
                        if [[ "$ERROR_MODE" == "exit" ]]; then
                            echo "错误模式为 exit，脚本即将退出..."
                            exit 1
                        else
                            record_failed_op "$dir_name"
                            continue
                        fi
                    }
                    new_op_name=mxrec_opp_"${dir_name}_310p".run
                    cd "$dir_name"
                    cp ./build_out/custom_opp*.run  "${new_op_name}"
                    mv "${new_op_name}" "${opp_output_path}"
                    _copy_plugin_dirs "${plugin_dir_names}"
                fi
            done
        elif [ "${BUILD_VER}" == "A3" ]; then
            for item in $support_A3_list; do
                if [ "$item" == "$dir_name" ]; then
                    if [ "${REBUILD_ALL}" == "false" ] && \
                        [ -f "${opp_output_path}"/mxrec_opp_"${dir_name}"_A3.run ]; then
                        echo "Operator ${dir_name} for A3 already built, skipping..."
                        continue
                    fi
                    bash ./run.sh --ai-core ai_core-Ascend910_93 || {
                        if [[ "$ERROR_MODE" == "exit" ]]; then
                            echo "Error mode is exit, the script will exit immediately..."
                            exit 1
                        else
                            record_failed_op "$dir_name"
                            continue
                        fi
                    }
                    new_op_name=mxrec_opp_"${dir_name}_A3".run
                    cd "$dir_name"
                    cp ./build_out/custom_opp*.run  "${new_op_name}"
                    mv "${new_op_name}" "${opp_output_path}"
                    _copy_plugin_dirs "${plugin_dir_names}"
                fi
            done
        elif [ "${BUILD_VER}" == "A2-TF" ]; then
            for item in $support_A2_tf_ops; do
                if [ "$item" == "$dir_name" ]; then
                    if [ "${REBUILD_ALL}" == "false" ] && \
                        [ -f "${opp_output_path}"/mxrec_opp_"${dir_name}".run ]; then
                        echo "Operator ${dir_name} for A2 already built, skipping..."
                        continue
                    fi
                    bash ./run.sh --ai-core ai_core-Ascend910B1 || {
                        if [[ "$ERROR_MODE" == "exit" ]]; then
                            echo "Error mode is exit, the script will exit immediately..."
                            exit 1
                        else
                            record_failed_op "$dir_name"
                            continue
                        fi
                    }
                    new_op_name=mxrec_opp_"${dir_name}".run
                    cd "$dir_name"
                    cp ./build_out/custom_opp*.run  "${new_op_name}"
                    mv "${new_op_name}" "${opp_output_path}"
                fi
            done
        elif [ "${BUILD_VER}" == "A2" ]; then
            in_list "$dir_name" $support_A2_tf_ops && exit 0
            if [ "${REBUILD_ALL}" == "false" ] && \
                [ -f "${opp_output_path}"/mxrec_opp_"${dir_name}".run ]; then
                echo "Operator ${dir_name} for A2 already built, skipping..."
                exit 0
            fi
            bash ./run.sh --ai-core ai_core-Ascend910B1 || {
                if [[ "$ERROR_MODE" == "exit" ]]; then
                    echo "Error mode is exit, the script will exit immediately..."
                    exit 1
                else
                    record_failed_op "$dir_name"
                    exit 0
                fi
            }
            new_op_name=mxrec_opp_"${dir_name}".run
            cd "$dir_name"
            cp ./build_out/custom_opp*.run  "${new_op_name}"
            mv "${new_op_name}" "${opp_output_path}"
            _copy_plugin_dirs "${plugin_dir_names}"
        elif [ "${BUILD_VER}" == "A5" ]; then
            if [ "${REBUILD_ALL}" == "false" ] && \
                [ -f "${opp_output_path}"/mxrec_opp_"${dir_name}".run ]; then
                echo "Operator ${dir_name} for A5 already built, skipping..."
                exit 0
            fi
            bash ./run.sh --ai-core ai_core-Ascend950 || {
                if [[ "$ERROR_MODE" == "exit" ]]; then
                    echo "Error mode is exit, the script will exit immediately..."
                    exit 1
                else
                    record_failed_op "$dir_name"
                    exit 0
                fi
            }
            new_op_name=mxrec_opp_"${dir_name}".run
            cd "$dir_name"
            cp ./build_out/custom_opp*.run  "${new_op_name}"
            mv "${new_op_name}" "${opp_output_path}"
            _copy_plugin_dirs "${plugin_dir_names}"
        fi
    ) > "${_log}" 2>&1 || _rc=$?
    _t1=$(date +%s)
    _elapsed=$((_t1 - _t0))
    printf '%4d\t%s\trc=%s\n' "${_elapsed}" "${dir_name}" "${_rc}" >> "${TIMING_FILE}"
    echo "[$(date +%H:%M:%S)] END   ${dir_name} rc=${_rc} elapsed=${_elapsed}s"

    # 只有「exit 模式编译失败 / cd 失败」时子 shell 才非零退出，此时补记失败
    if [ ${_rc} -ne 0 ]; then
        record_failed_op "${dir_name}"
    fi
    return "${_rc}"
}

# ---------------------------------------------------------------------------
# 终止本 shell 尚未结束的后台编译任务（ERROR_MODE=exit 早停用）。
# ---------------------------------------------------------------------------
function _kill_bg_jobs() {
    local pids
    pids=$(jobs -p)
    if [ -n "${pids}" ]; then
        kill ${pids} 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 并行执行算子列表（max_jobs 并发；max_jobs<=1 退化为原版串行）
# 用后台任务 + wait（而非 xargs bash -c）：后台任务 fork 会继承关联数组/函数等全部状态
# ---------------------------------------------------------------------------
function _run_ops_parallel() {
    local list_file="$1"
    local max_jobs="$2"
    local -a all=()
    local op
    while IFS= read -r op; do
        [ -z "$op" ] && continue
        all+=("$op")
    done < "${list_file}"

    if [ "${max_jobs}" -le 1 ] || [ "${#all[@]}" -le 1 ]; then
        for op in "${all[@]}"; do
            _compile_one_op "$op"
        done
        return 0
    fi

    # bash >= 4.3 支持 wait -n（动态队列，负载均衡最好）；否则回退到分批
    local use_wait_n=0
    if [ "${BASH_VERSINFO[0]}" -ge 5 ] || \
       { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 3 ]; }; then
        use_wait_n=1
    fi

    if [ "${use_wait_n}" -eq 1 ]; then
        local active=0
        local rc
        for op in "${all[@]}"; do
            _compile_one_op "$op" &
            active=$((active+1))
            if [ "${active}" -ge "${max_jobs}" ]; then
                rc=0
                wait -n || rc=$?
                active=$((active-1))
                if [ "${rc}" -ne 0 ]; then
                    _kill_bg_jobs
                    return 1
                fi
            fi
        done
        while [ "${active}" -gt 0 ]; do
            rc=0
            wait -n || rc=$?
            active=$((active-1))
            if [ "${rc}" -ne 0 ]; then
                _kill_bg_jobs
                return 1
            fi
        done
    else
        local -a batch=()
        local pid
        for op in "${all[@]}"; do
            batch+=("$op")
            if [ "${#batch[@]}" -ge "${max_jobs}" ]; then
                for b in "${batch[@]}"; do _compile_one_op "$b" & done
                for pid in $(jobs -p); do
                    wait "${pid}" || {
                        _kill_bg_jobs
                        return 1
                    }
                done
                batch=()
            fi
        done
        if [ "${#batch[@]}" -gt 0 ]; then
            for b in "${batch[@]}"; do _compile_one_op "$b" & done
            for pid in $(jobs -p); do
                wait "${pid}" || {
                    _kill_bg_jobs
                    return 1
                }
            done
        fi
    fi
    return 0
}

function compile_ops_v220() {
    echo "OP Path: $ops_path  (parallel=${PARALLEL_LEVEL})"
    local _list="${OPS_BUILD_LOG_DIR}/_op_list_v220.txt"
    _collect_ops "v220" "${ops_path}" "${_list}"
    echo "Operator count for v220: $(wc -l < "${_list}")"
    _run_ops_parallel "${_list}" "${PARALLEL_LEVEL}"
}

function compile_ops_A5() {
    echo "OP Path: $ops_path  (parallel=${PARALLEL_LEVEL})"
    local _list="${OPS_BUILD_LOG_DIR}/_op_list_a5.txt"
    _collect_ops "c310" "${ops_path}" "${_list}"
    echo "Operator count for A5/c310: $(wc -l < "${_list}")"
    _run_ops_parallel "${_list}" "${PARALLEL_LEVEL}"
}

function compile_ops() {
  if [ "${BUILD_VER}" == "A5" ]; then
      compile_ops_A5
  else
      compile_ops_v220
  fi
}

function get_tar_pkg() {
    cd "${CUR_DIR}"
    pkg_dir=recsdk-npu-ops
    release_tar=Ascend-"${pkg_dir}"-"${BUILD_VER}"-linux-"${ARCH}".tar.gz
    mkdir -p "${pkg_dir}"

    cp -r "${opp_output_path}" "${pkg_dir}"/
    cp -r "${output_path}/torch_plugin" "${pkg_dir}"/

    tar -zvcf "${release_tar}" "${pkg_dir}"
    rm -rf "${pkg_dir}"
    mv "${release_tar}" "${mxrec_output_path}"/"${release_tar}"
    echo "----------------------------------------------------"
    echo " Generate the file: "${mxrec_output_path}"/"${release_tar}" "
}


# start to build recsdk-npu-ops
setup_ccache
make_output_dir
echo "----------------          compile  custom ops for torchrec             ----------------"
compile_ops

cp_op_plugin

get_tar_pkg

# 汇总失败列表（从 FAILED_LOG 回填主进程数组）
failed_ops=()
if [ -s "${FAILED_LOG}" ]; then
    while IFS= read -r op; do
        [ -z "$op" ] && continue
        failed_ops+=("$op")
    done < "${FAILED_LOG}"
fi

if [ ${#failed_ops[@]} -ne 0 ]; then
    echo "Warning: The following operators failed to build:"
    for op in "${failed_ops[@]}"; do
        echo "- $op"
    done
    if [[ "${ERROR_MODE}" == "exit" ]]; then
        echo "ERROR_MODE=exit, fail the pipeline run"
        exit 1
    fi
else
    echo "All operators built successfully!"
    echo "----------------        compile success!!!!       ----------------"
fi

# 每个算子的编译耗时汇总（可选，便于观察并行收益）
if [ -s "${TIMING_FILE}" ]; then
    echo ""
    echo "================ per-operator compile time (top 10) ================"
    sort -rn "${TIMING_FILE}" | head -10
    echo "===================================================================="
fi

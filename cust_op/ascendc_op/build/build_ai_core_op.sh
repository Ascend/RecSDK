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
    echo "Error mode: ${ERROR_MODE}, the script will record the failed operator and continue to compile the remaining operators when an error occurs during compilation."
else
    echo "ERROR: Unknown value for ERROR_MODE: ${ERROR_MODE}, please set it to exit or continue."
    exit 1
fi

ARCH="$(uname -m)"
CUR_DIR=$(dirname "$(readlink -f "$0")")
ASCENDC_OP_DIR=$(dirname "${CUR_DIR}")
torch_plugin_path="${ASCENDC_OP_DIR}"/../framework/torch_plugin
ops_path="${ASCENDC_OP_DIR}"/ai_core_op
base_op_dir="v220"

source /etc/profile

declare -A OP_PLUGIN_MAP=(
  ["asynchronous_complete_cumsum"]="asynchronous_complete_cumsum"
  ["backward_codegen_adagrad_unweighted_exact"]="split_embedding_codegen_forward_unweighted"
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
  ["split_embedding_codegen_forward_unweighted"]="split_embedding_codegen_forward_unweighted"
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
split_embedding_codegen_forward_unweighted
backward_codegen_adagrad_unweighted_exact
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

failed_ops=()

function record_failed_op() {
    local op_name="$1"
    echo "Failed to build operator: ${op_name}"
    failed_ops+=("${op_name}")
}


function compile_ops_v220() {
    echo "OP Path: $ops_path"
    for dir in "$ops_path"/*; do
        cd "$ops_path"
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            plugin_dir_names=${OP_PLUGIN_MAP[$dir_name]}
            if [[ "$dir_name" == "cmake" || "$dir_name" == "common" || "$dir_name" == "custom_op_template" ]]; then
                continue
            fi
            cur_ver_op_dir=${dir_name}/${base_op_dir}
            if [ -d "$cur_ver_op_dir" ]; then
                echo "Entering directory: $dir_name, DIR: $dir"
                cd "$cur_ver_op_dir"
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
                            # copy torch_plugin
                            for plugin_dir_name in ${plugin_dir_names}; do
                              cp -r ${torch_plugin_path}/torch_library/${plugin_dir_name} "${plugin_output_path}"
                            done
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
                            for plugin_dir_name in ${plugin_dir_names}; do
                              cp -r ${torch_plugin_path}/torch_library/${plugin_dir_name} "${plugin_output_path}"
                            done
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
                    in_list "$dir_name" $support_A2_tf_ops && continue
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
                    for plugin_dir_name in ${plugin_dir_names}; do
                      cp -r ${torch_plugin_path}/torch_library/${plugin_dir_name} "${plugin_output_path}"
                    done
                fi
            fi
        fi
    done
}

function compile_ops_A5() {
    echo "OP Path: $ops_path"
    for dir in "$ops_path"/*; do
        cd "$ops_path"
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            plugin_dir_names=${OP_PLUGIN_MAP[$dir_name]}
            if [[ "$dir_name" == "cmake" || "$dir_name" == "common" ]]; then
                continue
            fi
            cur_ver_op_dir=${dir_name}/c310
            if [ -d "$cur_ver_op_dir" ]; then
                echo "Entering directory: $dir_name, DIR: $dir"
                cd "$cur_ver_op_dir"
                if [ "${REBUILD_ALL}" == "false" ] && \
                    [ -f "${opp_output_path}"/mxrec_opp_"${dir_name}".run ]; then
                    echo "Operator ${dir_name} for A5 already built, skipping..."
                    continue
                fi
                bash ./run.sh --ai-core ai_core-Ascend950 || { 
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
                for plugin_dir_name in ${plugin_dir_names}; do
                  cp -r ${torch_plugin_path}/torch_library/${plugin_dir_name} "${plugin_output_path}"
                done
            fi
        fi
    done
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
make_output_dir
echo "----------------          compile  custom ops for torchrec             ----------------"
compile_ops

cp_op_plugin

get_tar_pkg

if [ ${#failed_ops[@]} -ne 0 ]; then
    echo "Warning: The following operators failed to build:"
    for op in "${failed_ops[@]}"; do
        echo "- $op"
    done
else
    echo "All operators built successfully!"
    echo "----------------        compile success!!!!       ----------------"
fi

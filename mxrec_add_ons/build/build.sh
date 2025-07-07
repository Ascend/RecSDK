#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "${SCRIPT_DIR}")

source /etc/profile

torch_plugin_path="${MxRec_DIR}"/rec_for_torch/torch_plugin
ops_path="${MxRec_DIR}"/rec_for_torch/operators

support_A3_list="asynchronous_complete_cumsum
gather_for_rank1
index_select_for_rank1_backward
jagged_to_padded_dense
permute2d_sparse_data
split_embedding_codegen_forward_unweighted
dense_to_jagged
hstu_dense_forward_fuxi
hstu_dense_backward_fuxi
disetangle_attention
"
support_310p_list="gather_for_rank1
hstu_dense_forward_fuxi
relative_attn_bias_time
relative_attn_bias_pos
"

cd "${MxRec_DIR}"

function mv_op_plugin()
{
    cd "${torch_plugin_path}"
    mv ../torch_plugin "${output_path}"
}

function make_output_dir() {
    mxrec_output_path="${MxRec_DIR}"/output
    output_path="${SCRIPT_DIR}"/output
    opp_output_path="${SCRIPT_DIR}"/output/mxrec_ops
    mkdir -p "${mxrec_output_path}"
    mkdir -p "${output_path}"
    mkdir -p "${opp_output_path}"
}

function mv_doc_to_output() {
    cd "${MxRec_DIR}"/rec_for_torch
    mv ./docs "${output_path}"
}

function compile_ops() {
    echo "OP Path: $ops_path"
    for dir in "$ops_path"/*; do
        cd "$ops_path"
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            if [[ "$dir_name" == "cmake" || "$dir_name" == "common" ]]; then
                continue
            fi
            echo "Entering directory: $dir_name, DIR: $dir"
            cd "$dir_name"
            for item in $support_310p_list; do
                if [ "$item" == "$dir_name" ]; then
                    bash ./run.sh ai_core-Ascend310P3
                    new_op_name=mxrec_opp_"${dir_name}_310p".run
                    cd "$dir_name"
                    cp ./build_out/custom_opp*.run  "${new_op_name}"
                    mv "${new_op_name}" "${opp_output_path}"
                fi
            done
            cd "$ops_path"
            cd "$dir_name"
            for item in $support_A3_list; do
                if [ "$item" == "$dir_name" ]; then
                    bash ./run.sh ai_core-Ascend910_93
                    new_op_name=mxrec_opp_"${dir_name}_A3".run
                    cd "$dir_name"
                    cp ./build_out/custom_opp*.run  "${new_op_name}"
                    mv "${new_op_name}" "${opp_output_path}"
                fi
            done
            cd "$ops_path"
            cd "$dir_name"
            bash ./run.sh ai_core-Ascend910B1
            new_op_name=mxrec_opp_"${dir_name}".run
            cd "$dir_name"
            cp ./build_out/custom_opp*.run  "${new_op_name}"
            mv "${new_op_name}" "${opp_output_path}"
        fi
    done
}

function get_tar_pkg() {
    cd "${SCRIPT_DIR}"
    pkg_dir=mindxsdk-mxrec-add-ons
    release_tar=Ascend-"${pkg_dir}"-linux-"${ARCH}".tar.gz
    mkdir -p "${pkg_dir}"

    cp -r "${opp_output_path}" "${pkg_dir}"/

    cp -r "${output_path}"/torch_plugin "${pkg_dir}"/

    tar -zvcf "${release_tar}" "${pkg_dir}"
    rm -rf "${pkg_dir}"
    mv "${release_tar}" "${mxrec_output_path}"/"${release_tar}"
}


# start to build mxrec-add-ons
make_output_dir
echo "----------------          compile  custom ops for torchrec             ----------------"
compile_ops

mv_op_plugin

get_tar_pkg
echo "----------------        compile success!!!!       ----------------"

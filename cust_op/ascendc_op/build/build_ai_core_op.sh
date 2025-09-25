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

if [ "$#" -ne 1 ]; then
    echo "ERROR: Please specify the version to compile. e.g. 'bash $0 v220'"
    exit 1
fi

BUILD_VER=${1,,} # To lowercase

if [ "${BUILD_VER}" == "v220" ]; then
    echo "BUILD_VER: ${BUILD_VER}"
else
    echo "ERROR: Unknown BUILD_VER:${BUILD_VER}"
    exit 1
fi

ARCH="$(uname -m)"
CUR_DIR=$(dirname "$(readlink -f "$0")")
ASCENDC_OP_DIR=$(dirname "${CUR_DIR}")
torch_plugin_path="${ASCENDC_OP_DIR}"/../framework/torch_plugin
ops_path="${ASCENDC_OP_DIR}"/ai_core_op

source /etc/profile


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

cd "${ASCENDC_OP_DIR}"

function cp_op_plugin()
{
    cd "${torch_plugin_path}"
    cp -r ../torch_plugin "${output_path}"
}

function make_output_dir() {
    mxrec_output_path="${ASCENDC_OP_DIR}"/output
    output_path="${CUR_DIR}"/output
    opp_output_path="${CUR_DIR}"/output/recsdk_ops
    mkdir -p "${mxrec_output_path}"
    mkdir -p "${output_path}"
    mkdir -p "${opp_output_path}"
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
            cur_ver_op_dir=${dir_name}/${BUILD_VER}
            echo "Entering directory: $dir_name, DIR: $dir"
            cd "$cur_ver_op_dir"
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
            cd "$cur_ver_op_dir"
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
            cd "$cur_ver_op_dir"
            bash ./run.sh ai_core-Ascend910B1
            new_op_name=mxrec_opp_"${dir_name}".run
            cd "$dir_name"
            cp ./build_out/custom_opp*.run  "${new_op_name}"
            mv "${new_op_name}" "${opp_output_path}"
        fi
    done
}

function get_tar_pkg() {
    cd "${CUR_DIR}"
    pkg_dir=recsdk-npu-ops
    release_tar=Ascend-"${pkg_dir}"-"${BUILD_VER}"-linux-"${ARCH}".tar.gz
    mkdir -p "${pkg_dir}"

    cp -r "${opp_output_path}" "${pkg_dir}"/

    cp -r "${output_path}"/torch_plugin "${pkg_dir}"/

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
echo "----------------        compile success!!!!       ----------------"

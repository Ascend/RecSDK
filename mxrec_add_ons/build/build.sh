#!/bin/bash
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

##################################################################
#   build.sh 编译torch plugin、 自定义算子
# 编译环境：Python3.9.6 GCC 10.2.1 CMake 3.20.6
# 代码主要分为两部分：
# 1、编译torch plugin
# 2、编译自定义算子
##################################################################

set -e
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "${SCRIPT_DIR}")

source /etc/profile

torch_plugin_path="${MxRec_DIR}"/rec_for_torch/torch-plugin
tf_plugin_path="${MxRec_DIR}"/rec_for_torch/tf_plugin
onnx_plugin_path="${MxRec_DIR}"/rec_for_torch/onnx_plugin
ops_path="${MxRec_DIR}"/rec_for_torch/operators
gr_path="${MxRec_DIR}"/rec_for_torch/generative-recommenders
torch_rec_path="${MxRec_DIR}"/rec_for_torch/torchrec
hstu_demo_path="${MxRec_DIR}"/rec_for_torch/demo_script

support_A3_list="asynchronous_complete_cumsum
dense_to_jagged
jagged_to_padded_dense
hstu_dense_forward
hstu_dense_backward
index_select_for_rank1_backward
gather_for_rank1
backward_codegen_adagrad_unweighted_exact
permute2d_sparse_data
split_embedding_codegen_forward_unweighted
bounds_check_indices
"
support_310p_list="asynchronous_complete_cumsum dense_to_jagged jagged_to_padded_dense hstu_dense_forward gather_for_rank1"

cd "${MxRec_DIR}"

function compile_torch_plugin()
{
    cd "${torch_plugin_path}"/torch_library
    bash package.sh
    cd "${torch_plugin_path}"
    bash run_op_plugin.sh
}

function mv_whl_to_output()
{
    cd "${torch_plugin_path}"
    mv ./op-plugin/dist/*.whl "${output_path}"
}

function mv_op_plugin()
{
    cd "${tf_plugin_path}"
    mv ./tf_ops "${output_path}"

    cd "${torch_plugin_path}"
    mv ./torch_library "${output_path}"
}

function mv_example_to_output()
{
    cd "${tf_plugin_path}"
    mv ./tf_demo "${output_path}"

    cd "${torch_plugin_path}"
    mv ./torch_demo "${output_path}"

    cd "${onnx_plugin_path}"
    mv ./onnx_demo "${output_path}"
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
            if [[ "$dir_name" == "cmake" ]]; then
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

function get_poc_tar_pkg() {
    cd "${SCRIPT_DIR}"
    pkg_dir=mindxsdk-mxrec-add-ons-poc
    release_tar=Ascend-"${pkg_dir}"-linux-"${ARCH}".tar.gz
    mkdir -p "${pkg_dir}"
    mkdir -p "${pkg_dir}"/torch_plugin
    mkdir -p "${pkg_dir}"/generative-recommenders
    mkdir -p "${pkg_dir}"/torchrec
    mkdir -p "${pkg_dir}"/example
    mkdir -p "${pkg_dir}"/torch_library

    cp "${output_path}"/*.whl "${pkg_dir}"/torch_plugin
    cp -r "${opp_output_path}" "${pkg_dir}"/
    cp "${gr_path}"/*.md "${pkg_dir}"/generative-recommenders
    torchrec_list="torchrec_dlrm.patch Dockerfile Dockerfile_debian hybrid_dlrm.md"
    for files in $torchrec_list; do
        if [ -e "${torch_rec_path}"/"${files}" ]; then
            cp "${torch_rec_path}"/"${files}" "${pkg_dir}"/torchrec
        fi
    done
    cp -r "${torch_rec_path}"/parallel_hashmap "${pkg_dir}"/torchrec
    cp -r "${output_path}"/tf_demo "${pkg_dir}"/example
    cp -r "${output_path}"/torch_demo "${pkg_dir}"/example
    cp -r "${output_path}"/onnx_demo "${pkg_dir}"/example
    cp -r "${output_path}"/torch_library/output/. "${pkg_dir}"/torch_library
    cp -r "${output_path}"/tf_ops "${pkg_dir}"
    cp -r "${output_path}"/docs "${pkg_dir}"/
    cp "${hstu_demo_path}"/test_hstu.sh "${pkg_dir}"/example

    tar -zvcf "${release_tar}" "${pkg_dir}"
    rm -rf "${pkg_dir}"
    mv "${release_tar}" "${mxrec_output_path}"/"${release_tar}"
}

function get_commercial_tar_pkg() {
    echo "----------------          tar commerical pkg             ----------------"
    commercial_op_list="hstu_dense_forward hstu_dense_backward"
    commercial_torch_library_list="hstu common"
    commercial_docs_list="hstu_forward hstu_backward"
    commercial_demo_list="hstu_dense"

    cd "${SCRIPT_DIR}"
    pkg_dir=mindxsdk-mxrec-add-ons
    release_tar=Ascend-"${pkg_dir}"-linux-"${ARCH}".tar.gz

    mkdir -p "${pkg_dir}"
    mkdir -p "${pkg_dir}/docs"
    mkdir -p "${pkg_dir}/example/torch_demo"
    mkdir -p "${pkg_dir}/example/tf_demo"
    mkdir -p "${pkg_dir}/mxrec_ops"
    mkdir -p "${pkg_dir}/torch_library"

    # cp ops
    for ops in $commercial_op_list; do
        op_run_name=mxrec_opp_"${ops}".run
        cp ${opp_output_path}/${op_run_name} "${pkg_dir}/mxrec_ops/"
    done

    # cp op plugin
    for ops in $commercial_torch_library_list; do
        cp -r "${output_path}/torch_library/2.1.0/${ops}" "${pkg_dir}/torch_library/"
    done

    # cp tf plugin
    cp -r "${output_path}/tf_ops" "${pkg_dir}"

    # cp op docs
    for ops in $commercial_docs_list; do
        docs_name="${ops}.md"
        cp "${output_path}/docs/${docs_name}" "${pkg_dir}/docs/"
    done

    # cp op example
    for dir in $commercial_demo_list; do
        ops_demo_name="${dir}"
        cp -r "${output_path}/torch_demo/${dir}" "${pkg_dir}/example/torch_demo/"
    done
    cp -r "${output_path}"/tf_demo "${pkg_dir}"/example

    cp "${hstu_demo_path}"/test_hstu.sh "${pkg_dir}"/example

    tar -zvcf "${release_tar}" "${pkg_dir}"
    rm -rf "${pkg_dir}"
    mv "${release_tar}" "${mxrec_output_path}"/"${release_tar}"
}


# start to build mxrec-add-ons
make_output_dir
echo "----------------          compile  custom ops for torchrec             ----------------"
compile_ops
echo "----------------          compile  torch plugin           ----------------"
compile_torch_plugin
mv_whl_to_output
echo "----------------          compile  tf plugin           ----------------"
mv_op_plugin
mv_example_to_output
mv_doc_to_output
get_poc_tar_pkg
get_commercial_tar_pkg
echo "----------------        compile success!!!!       ----------------"

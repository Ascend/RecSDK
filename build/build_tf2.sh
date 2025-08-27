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
#   build_tf2.sh 编译Rec SDK
# 编译环境：Python3.7.5 GCC 7.3.0 CMake 3.20.6
# 代码主要分为两部分：
# 1、准备编译Rec SDK所需依赖：pybind11(v2.10.3) securec
# 2、编译securec、AccCTR以及Rec SDK
##################################################################

set -e
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "${SCRIPT_DIR}")

export CC=$(which gcc)

opensource_path="${MxRec_DIR}"/../opensource
if [ ! -d ${opensource_path} ]; then
  echo "user should download dependency packages to mxRec/../opensource directory, see README.md"
  exit -1
fi

function prepare_pybind(){
  cd "${opensource_path}"
  if [ ! -d pybind11 ]; then
    unzip pybind11-2.10.3.zip
    mv pybind11-2.10.3 pybind11
  fi
}

function prepare_securec(){
  cd "${opensource_path}"
  if [ ! -d securec ]; then
    unzip huaweicloud-sdk-c-obs-3.23.9.zip
    mv huaweicloud-sdk-c-obs-3.23.9/platform/huaweisecurec securec
    rm -rf huaweicloud-sdk-c-obs-3.23.9
    rm -rf securec/lib/*
  fi
}

# 准备pybind11和securec
echo "opensource path:${opensource_path}"
prepare_pybind
prepare_securec

# 配置tf2路径
[ -e /opt/buildtools/tf2_env/bin/activate ] && source /opt/buildtools/tf2_env/bin/activate
tf2_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/tensorflow
[ -e /opt/buildtools/tf2_env/bin/activate ] && deactivate tf2_env

# 配置Rec SDK C++代码路径和AccCTR路径
cust_op_tf_plugin_path="${MxRec_DIR}"/cust_op/framework/tf_plugin
common_src_path="${MxRec_DIR}"/training/common/src
common_python_path="${MxRec_DIR}"/training/common/python
tf_rec_v1_path="${MxRec_DIR}"/training/tf_rec_v1/python
tf_rec_v1_src_path="${MxRec_DIR}"/training/tf_rec_v1/src
acc_ctr_path="${tf_rec_v1_src_path}"/AccCTR
cd "${MxRec_DIR}"

function compile_securec()
{
    if [[ ! -d "${opensource_path}"/securec ]]; then
      echo "securec is not exist"
      exit 1
    fi

    if [[ ! -f "${opensource_path}"/securec/lib/libsecurec.so ]]; then
      cd "${opensource_path}"/securec/src
      make -j4
    fi
}

function compile_recsdk_tf_npu_ops_tf_plugin_so_file()
{
  cd "${cust_op_tf_plugin_path}"
  chmod u+x build.sh
  ./build.sh "$1" "${MxRec_DIR}"
  cd ${MxRec_DIR}
}

function compile_tf_rec_v1_so_file()
{
  cd "${tf_rec_v1_src_path}"
  chmod u+x build.sh
  ./build.sh "$1" "${MxRec_DIR}" "YES"
  cd ..
}

function compile_acc_ctr_so_file()
{
  cd "${acc_ctr_path}"
  chmod u+x build.sh
  ./build.sh "release"
}

function compile_common_so_file()
{
    cd "${common_src_path}"
    chmod u+x build.sh
    ./build.sh "$1" "${MxRec_DIR}" "YES"
}

function collect_so_file()
{
  cd "${common_src_path}"
  rm -rf "${common_src_path}"/lib
  mkdir -p "${common_src_path}"/lib
  chmod u+x lib
  cp "${common_src_path}"/build/pybind/*.so ./lib
  cp "${common_src_path}"/build/core/*.so ./lib
  rm -rf "${common_python_path}"/lib
  mv "${common_src_path}"/lib "${common_python_path}"
  touch "${common_python_path}"/lib/__init__.py

  cd "${tf_rec_v1_src_path}"
  rm -rf "${tf_rec_v1_src_path}"/libasc
  mkdir -p "${tf_rec_v1_src_path}"/libasc
  chmod u+x libasc

  cp ${acc_ctr_path}/output/ock_ctr_common/lib/* libasc
  cp -df "${MxRec_DIR}"/tf_rec_v1_output/*.so* libasc
  cp -df "${MxRec_DIR}"/cust_op_output/*.so* libasc
  cp "${opensource_path}"/securec/lib/libsecurec.so libasc
  cd "${MxRec_DIR}"
  touch "${tf_rec_v1_src_path}"/libasc/__init__.py
  rm -rf "${tf_rec_v1_path}"/libasc
  mv "${tf_rec_v1_src_path}"/libasc "${tf_rec_v1_path}"
}

# start to build Rec SDK
echo "----------------          compile     securec           ----------------"
compile_securec
echo "----------------          compile common so files       ----------------"
compile_common_so_file "${tf2_path}"
echo "----------------          compile     AccCTR            ----------------"
compile_acc_ctr_so_file
echo "----------------          compile MxRec so files        ----------------"
compile_recsdk_tf_npu_ops_tf_plugin_so_file "${tf2_path}"
compile_tf_rec_v1_so_file "${tf2_path}"
echo "---------------- collect so files and mv them to libasc ----------------"
collect_so_file
echo "----------------        compile MxRec success!!!!       ----------------"

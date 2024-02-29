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
#   build_tf2_with_opensource.sh 编译MxRec和动态扩容算子
# 编译环境：Python3.7.5 GCC 7.3.0 CMake 3.20.6
# 代码主要分为四部分：
# 1、准备编译MxRec所需依赖：pybind11(v2.10.3) securec
# 2、编译securec、AccCTR以及MxRec
# 3、生成MxRec Wheel包，生成的whl包在当前目录下的mindxsdk-mxrec/tf2_whl
# 4、编译动态扩容算子
##################################################################

set -e
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "${SCRIPT_DIR}")

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
tf2_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/tensorflow

project_output_path="${MxRec_DIR}"/output/
VERSION_FILE="${MxRec_DIR}"/../mindxsdk/build/conf/config.yaml

function get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed '/.*mindxsdk:/!d;s/.*: //' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="5.0.0"
  fi
}

rm -rf  "${project_output_path}"
rm -rf  "${SCRIPT_DIR}/lib"

# 获取MxRec版本信息
get_version
export VERSION
echo "MindX SDK MxRec: ${VERSION}" >> ./version.info

pkg_dir=mindxsdk-mxrec
rm -rf "${pkg_dir}"
mkdir "${pkg_dir}"
mv version.info "${pkg_dir}"

# 配置MxRec C++代码路径和AccCTR路径
src_path="${MxRec_DIR}"/src
acc_ctr_path="${MxRec_DIR}"/src/AccCTR
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

function compile_so_file()
{
  cd "${src_path}"
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

function collect_so_file()
{
  cd "${src_path}"
  rm -rf "${src_path}"/libasc
  mkdir -p "${src_path}"/libasc
  chmod u+x libasc

  cp ${acc_ctr_path}/output/ock_ctr_common/lib/* libasc
  cp -df "${MxRec_DIR}"/output/*.so* libasc
  cp "${opensource_path}"/securec/lib/libsecurec.so libasc
}

function gen_wheel_file()
{
  cd "${MxRec_DIR}"
  touch "${src_path}"/libasc/__init__.py
  rm -rf "${MxRec_DIR}"/mx_rec/libasc
  mv "${src_path}"/libasc "${MxRec_DIR}"/mx_rec
  python3.7 setup.py bdist_wheel --plat-name=linux_$(arch)
  mkdir -p "$1"
  echo "moving whl file $1"
  mv dist/mx_rec*.whl "$1"
  rm -rf "${MxRec_DIR}"/mx_rec/libasc
}

# start to build MxRec
echo "----------------          compile     securec           ----------------"
compile_securec
echo "----------------          compile     AccCTR            ----------------"
compile_acc_ctr_so_file
echo "----------------          compile MxRec so files        ----------------"
compile_so_file "${tf2_path}"
echo "---------------- collect so files and mv them to libasc ----------------"
collect_so_file
echo "----------------      generate MxRec wheel package      ----------------"
gen_wheel_file  "$SCRIPT_DIR"/"${pkg_dir}"/tf2_whl
echo "----------------        compile MxRec success!!!!       ----------------"

# start to compile cust op
echo "----------------        start to compile cust op        ----------------"
cd "${MxRec_DIR}"/cust_op/cust_op_by_addr
chmod u+x run.sh
./run.sh
echo "----------------      compile cust op success!!!!       ----------------"
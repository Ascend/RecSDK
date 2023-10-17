#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
# Description: build script.
# Author: MindX SDK
# Create: 2023
# History: NA

##################################################################
#   build_tf1_with_opensource.sh 用于美团客户编译MxRec和动态扩容算子
# 编译环境：Python3.7.5 GCC 7.3.0 CMake 3.20.6
# 代码主要分为四部分：
# 1、准备编译MxRec所需依赖：pybind11(v2.10.3) securec
# 2、编译securec、AccCTR以及MxRec
# 3、生成MxRec Wheel包，生成的whl包在当前目录下的mindxsdk-mxrec/tf1_whl
# 4、编译动态扩容算子
##################################################################

set -e
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "${SCRIPT_DIR}")

opensource_path="${MxRec_DIR}"/opensource

function prepare_pybind_and_securec() {
  if [ ! -d pybind11 ]; then
    if [ ! -d pybind11-v2.10.3 ]; then
      unzip pybind11-v2.10.3.zip
    fi
    mv pybind11-v2.10.3 pybind11
  fi

  if [ ! -d glog ]; then
    if [ ! -d glog-0.6.0 ]; then
      tar -zxvf glog-0.6.0.tar.gz
    fi
    mv glog-0.6.0 glog
  fi

  if [ ! -d securec ]; then
    unzip securec.zip
    rm -rf securec/lib/*
    if [ ! -d ../platform ]; then
      mkdir -p ../platform
      cp -rf securec ../platform
    fi
  fi
}

# 准备pybind11和securec
cd "${opensource_path}"
prepare_pybind_and_securec
cd -

# 配置tf1路径
tf1_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/tensorflow_core

project_output_path="${MxRec_DIR}"/output/
VERSION_FILE="${MxRec_DIR}"/../mindxsdk/build/conf/config.yaml

function get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed '/.*mindxsdk:/!d;s/.*: //' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="5.0.rc3"
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
acc_ctr_path="${MxRec_DIR}"/src/platform/AccCTR
cp -rf "${MxRec_DIR}"/platform/securec/* "${acc_ctr_path}"/3rdparty/huawei_secure_c
cd "${MxRec_DIR}"

function compile_securec()
{
    if [[ ! -d "${MxRec_DIR}"/platform/securec ]]; then
        echo "securec is not exist"
        exit 1
    fi

    if [[ ! -f "${MxRec_DIR}"/platform/securec/lib/libsecurec.so ]]; then
        cd "${MxRec_DIR}"/platform/securec/src
        make -j
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
  cp "${MxRec_DIR}"/platform/securec/lib/libsecurec.so libasc
}

function gen_wheel_file()
{
  cd "${MxRec_DIR}"
  touch "${src_path}"/libasc/__init__.py
  rm -rf "${MxRec_DIR}"/mx_rec/libasc
  mv "${src_path}"/libasc "${MxRec_DIR}"/mx_rec
  python3.7 setup.py bdist_wheel --plat-name=linux_$(arch)
  mkdir -p "$1"
  mv dist/mx_rec*.whl "$1"
  rm -rf "${MxRec_DIR}"/mx_rec/libasc
}

# start to build MxRec
echo "----------------          compile     securec           ----------------"
compile_securec
echo "----------------          compile     AccCTR            ----------------"
compile_acc_ctr_so_file
echo "----------------          compile MxRec so files        ----------------"
compile_so_file "${tf1_path}"
echo "---------------- collect so files and mv them to libasc ----------------"
collect_so_file
echo "----------------      generate MxRec wheel package      ----------------"
gen_wheel_file  "$SCRIPT_DIR"/"${pkg_dir}"/tf1_whl
echo "----------------        compile MxRec success!!!!       ----------------"

# start to compile cust op
echo "----------------        start to compile cust op        ----------------"
cd "${MxRec_DIR}"/cust_op/cust_op_by_addr
chmod u+x run.sh
./run.sh
echo "----------------      compile cust op success!!!!       ----------------"
#!/bin/bash
# -*- coding: utf-8 -*-
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
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT_DIR=$(dirname "$(dirname "${SCRIPT_DIR}")")

MxRec_DIR="${PROJECT_ROOT_DIR}"/../../
common_src_path="${MxRec_DIR}"/training/common/src
common_python_path="${MxRec_DIR}"/training/common/python
opensource_path="${MxRec_DIR}"/../opensource

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

function compile_common_so_file()
{
    cd "${common_src_path}"
    chmod u+x build.sh
    sed -i 's/which python3.7/which python3.11/g' build.sh
    CMAKE_FILE="${common_src_path}/CMakeLists.txt"
    sed -i 's|set(PYTHON_INCLUDE_DIR ${PYTHON_PATH}/include/python3.7)|set(PYTHON_INCLUDE_DIR ${PYTHON_PATH}/include/python3.11)|g' "${CMAKE_FILE}"
    sed -i 's|set(PYTHON_LIBRARY ${PYTHON_PATH}/lib/libpython3.7m.so)|set(PYTHON_LIBRARY ${PYTHON_PATH}/lib/libpython3.11.so)|g' "${CMAKE_FILE}"
    sed -i 's/find_package(PythonLibs 3.7 REQUIRED)/find_package(PythonLibs 3.11 REQUIRED)/g' "${CMAKE_FILE}"
    ./build.sh "${MxRec_DIR}" "YES"
}

function collect_common_so_file()
{
  cd "${common_src_path}"
  rm -rf "${common_src_path}"/lib
  mkdir -p "${common_src_path}"/lib
  chmod u+x lib
  cp "${common_src_path}"/build/pybind/*.so ./lib
  cp "${common_src_path}"/build/core/*.so ./lib
  cp "${opensource_path}"/securec/lib/libsecurec.so ./lib
  rm -rf "${common_python_path}"/lib
  mv "${common_src_path}"/lib "${common_python_path}"
  touch "${common_python_path}"/lib/__init__.py
  chmod 640 "${common_python_path}"/lib/__init__.py
}

echo "----------------        moving files to new structure   ----------------"
compile_securec
compile_common_so_file
collect_common_so_file
echo "------------------        compile success!!!!      ---------------------"

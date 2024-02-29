#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# build 3rdparty
set -e
CURRENT_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
BUILD_MODE=$1
INSTALL_PATH=${CURRENT_PATH}/../install
OPENSOURCE_PATH=${CURRENT_PATH}/../../../../opensource

if [ ! -d "${INSTALL_PATH}" ]; then
  mkdir -p ${INSTALL_PATH}
else
  rm -rf ${INSTALL_PATH}/*
fi

GTEST_SRC_PATH=${OPENSOURCE_PATH}/googletest-release-1.8.1
echo "${GTEST_SRC_PATH}"
GTEST_INSTALL_PATH=${INSTALL_PATH}/googletest-release-1.8.1

install_gtest() {
  [ -n "${GTEST_INSTALL_PATH}" ] && rm -rf "${GTEST_INSTALL_PATH}"
  echo "${GTEST_INSTALL_PATH}"
  if [[ ! -d "${GTEST_INSTALL_PATH}" ]]; then
    mkdir -p "${GTEST_INSTALL_PATH}"
  fi

  cd "${GTEST_SRC_PATH}"
  echo "${GTEST_SRC_PATH}"
  cmake -DCMAKE_INSTALL_PREFIX="${GTEST_INSTALL_PATH}" -DCMAKE_INSTALL_LIBDIR=lib64 . && make && make install
}

function prepare_securec(){
  cd ${OPENSOURCE_PATH}
  if [ ! -d securec ]; then
    unzip huaweicloud-sdk-c-obs-3.23.9.zip
    mv huaweicloud-sdk-c-obs-3.23.9/platform/huaweisecurec securec
    rm -rf huaweicloud-sdk-c-obs-3.23.9
    rm -rf securec/lib/*
  fi
}

prepare_securec
SECUREC_SRC_PATH=${OPENSOURCE_PATH}/securec/src
echo "${SECUREC_SRC_PATH}"
SECUREC_INSTALL_PATH=${INSTALL_PATH}/securec
compile_securec() {
  [ -n "${SECUREC_INSTALL_PATH}" ] && rm -rf "${SECUREC_INSTALL_PATH}"
  echo "${SECUREC_INSTALL_PATH}"
  if [[ ! -d "${SECUREC_INSTALL_PATH}" ]]; then
    mkdir -p "${SECUREC_INSTALL_PATH}"
  fi
  cd "${SECUREC_SRC_PATH}"
  make -j
  scp -r ${SECUREC_SRC_PATH}/../include ${SECUREC_INSTALL_PATH}/
  scp -r ${SECUREC_SRC_PATH}/../lib ${SECUREC_INSTALL_PATH}/
}


if [[ "${BUILD_MODE}" == "ut" ]];then
    BUILD_MODE="debug"
    install_gtest
    echo "compiled GTest"
fi

compile_securec
echo "compiled huawei securec"
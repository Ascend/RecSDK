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

# build src
set -e
CURRENT_PATH=$(dirname "$0")
BUILD_PATH=${CURRENT_PATH}/src
CPU_TYPE=$(arch)
BUILD_MODE=$1
echo "${BUILD_PATH}"
if [ ! -d "${BUILD_PATH}" ]; then
  mkdir -p ${BUILD_PATH}
else
  rm -rf ${BUILD_PATH}/*
fi

source ${CURRENT_PATH}/build_env.sh
cd ${BUILD_PATH}
cmake ${OCK_CTR_PATH} -DCMAKE_INSTALL_PREFIX:STRING=${OUTPUT_PATH}/ock_ctr_common -DCTR_ENV=${CPU_TYPE} -DBUILD_MODE=${BUILD_MODE}

if [ 0 != $? ];then
    echo "cmake failed."
	exit 1
fi
echo "cmake success."

make clean; make -j 4; make install;

if [ 0 != $? ];then
    echo "make failed."
	exit 1
fi
echo "make success."

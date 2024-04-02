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

# Used for memory pool kit builds
# BUILD_TYPE 1、release 2、ut 3、perf
set -ex
readonly CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
BUILD_TYPE=$1

TOP_DIR=${CURRENT_PATH}
OUTPUT_PATH=${TOP_DIR}/output
OCK_CTR_PATH=${TOP_DIR}
OCK_CTR_DIAGNOSE_PATH=${TOP_DIR}/tests
BUILD_PATH=${CURRENT_PATH}/build

# default is use build Release version
if [ "${BUILD_TYPE}" == "debug" ];then
    BUILD_MODE="debug"
elif [ "${BUILD_TYPE}" == "ut" ];then
    BUILD_MODE="ut"
else
    BUILD_MODE="release"
fi

cd ${BUILD_PATH}
cmake ${OCK_CTR_PATH} -DCMAKE_INSTALL_PREFIX:STRING=${OUTPUT_PATH}/ock_ctr_common -DCTR_ENV=${CPU_TYPE} -DBUILD_MODE=${BUILD_MODE}
if [ 0 != $? ];then
      echo "Failed to build_src"
      exit 1
fi

make clean; make -j 4; make install;
if [ 0 != $? ];then
      echo "Failed to build_src"
      exit 1
fi
cd -

if [[ "${BUILD_TYPE}" == "ut" ]];then
    cp ${CURRENT_PATH}/../../../opensource/securec/lib/libsecurec.so ${CURRENT_PATH}/output/ock_ctr_common/lib/
    export LD_LIBRARY_PATH=${CURRENT_PATH}/output/ock_ctr_common/lib:$LD_LIBRARY_PATH
fi

echo "build end!"
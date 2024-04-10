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

# build pkg
set -e
readonly CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
PKG_PATH=${CURRENT_PATH}/../pkg
OUTPUT_PATH=${CURRENT_PATH}/../output
if [ ! -d "${OUTPUT_PATH}" ]; then
  echo "${OUTPUT_PATH} not exist, user should call build_src.sh first"
fi

INSTALL_PATH=${CURRENT_PATH}/../install

echo "${PKG_PATH}"
if [ ! -d "${PKG_PATH}" ]; then
  mkdir -p ${PKG_PATH}
else
  rm -rf ${PKG_PATH}/*
fi

scp -r ${OUTPUT_PATH}/* ${PKG_PATH}/
scp -r ${INSTALL_PATH}/securec/include/* ${PKG_PATH}/ock_ctr_common/include/
scp -r ${INSTALL_PATH}/securec/lib/* ${PKG_PATH}/ock_ctr_common/lib/






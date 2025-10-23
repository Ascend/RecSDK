#!/bin/bash
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

export CC=/usr/local/openmpi/bin/mpicc

cd ./src
rm -rf ./build
mkdir ./build
cd ./build

ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

tf1_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/tensorflow_core
tfa_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/npu_bridge
mx_rec_package_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/mx_rec
so_path=${mx_rec_package_path}/libasc

export LD_PRELOAD=/lib64/libgomp.so.1
export LD_LIBRARY_PATH=${so_path}:{tfa_path}:/usr/local/lib:$LD_LIBRARY_PATH

python_path="$(dirname "$(dirname "$(realpath "$(which python3.7)")")")"
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    ascend_path=/usr/local/Ascend/ascend-toolkit/latest
elif [ -d /usr/local/Ascend/latest ]; then
    ascend_path=/usr/local/Ascend/latest
else
    echo "ERROR: can not find toolkit and tfplugin"
    exit 1
fi
echo "SCRIPT_DIR = " ${SCRIPT_DIR}
pwd
MxRec_DIR=$(dirname "${SCRIPT_DIR}")/../../../../..
echo "MxRec_DIR = " $MxRec_DIR

opensource_path="${MxRec_DIR}"/../opensource
echo $opensource_path

echo "=============start build host============="
cmake -DCMAKE_BUILD_TYPE=Release \
      -DASCEND_PATH="$ascend_path" \
      -DTF_PATH="$tf1_path" \
      -DABSEIL_PATH="$tf1_path" \
      -DOMPI_PATH="$(whereis openmpi)" \
      -DPYTHON_PATH="$python_path" \
      -DSECUREC_PATH="$opensource_path"/securec \
      -DOPENSOURCE_DIR="$opensource_path"  ..
make -j4
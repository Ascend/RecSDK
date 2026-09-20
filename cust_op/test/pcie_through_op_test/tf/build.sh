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
# locate openmpi prefix: prefer OpenMPI's own query, then mpicc-derived path,
# then common install dirs (source build /usr/local/openmpi, yum /usr/lib64/openmpi)
ompi_path=""
if command -v mpirun >/dev/null 2>&1; then
    ompi_path="$(mpirun --showme:prefix 2>/dev/null || true)"
fi
if [ -z "$ompi_path" ] && command -v mpicc >/dev/null 2>&1; then
    ompi_path="$(dirname "$(dirname "$(readlink -f "$(command -v mpicc)")")")"
fi
if [ -n "$ompi_path" ] && [ ! -f "$ompi_path/include/mpi.h" ] && [ ! -x "$ompi_path/bin/mpicc" ]; then
    ompi_path=""
fi
if [ -z "$ompi_path" ]; then
    for candidate in /usr/local/openmpi /usr/lib64/openmpi; do
        if [ -f "$candidate/include/mpi.h" ] || [ -x "$candidate/bin/mpicc" ]; then
            ompi_path="$candidate"
            break
        fi
    done
fi
if [ -z "$ompi_path" ]; then
    echo "ERROR: openmpi not found"
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
      -DOMPI_PATH="$ompi_path" \
      -DPYTHON_PATH="$python_path" \
      -DSECUREC_PATH="$opensource_path"/securec \
      -DOPENSOURCE_DIR="$opensource_path"  ..
make -j4

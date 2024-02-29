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

set -e
[ -d build ] && rm -rf build;
mkdir build && cd build || exit 1
# HDF5_PATH, EASY_PROFILER_PATH is optional
python_path="$(dirname "$(dirname "$(realpath "$(which python3.7)")")")"
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    ascend_path=/usr/local/Ascend/ascend-toolkit/latest
elif [ -d /usr/local/Ascend/latest ]; then
    ascend_path=/usr/local/Ascend/latest
else
    echo "ERROR: can not find toolkit and tfplugin"
    exit 1
fi

cmake -DCMAKE_BUILD_TYPE=Release \
    -DTF_PATH="$1" \
    -DOMPI_PATH="$(whereis openmpi)" \
    -DPYTHON_PATH="$python_path" \
    -DEASY_PROFILER_PATH=/ \
    -DASCEND_PATH="$ascend_path" \
    -DABSEIL_PATH="$1" \
    -DSECUREC_PATH="$2"/../opensource/securec \
    -DCMAKE_INSTALL_PREFIX="$2"/output \
    -DBUILD_CUST="$3" ..
make -j
make install

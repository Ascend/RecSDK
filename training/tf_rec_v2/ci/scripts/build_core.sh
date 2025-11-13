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

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <python_home> <tf_version>"
    exit 1
fi

python_home=$1
tf_version=$2
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    ascend_toolkit_home=/usr/local/Ascend/ascend-toolkit/latest
elif [ -d /usr/local/Ascend/latest ]; then
    ascend_toolkit_home=/usr/local/Ascend/latest
else
    echo "ERROR: can not find toolkit and tfplugin"
    exit 1
fi

scripts_dir=$(dirname "$(readlink -f "$0")")
project_dir=$(dirname "$(dirname "${scripts_dir}")")
build_dir=$project_dir/build

cmake -S "$project_dir" -B $build_dir \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DPYTHON_HOME=$python_home \
    -DASCEND_TOOLKIT_HOME=$ascend_toolkit_home \
    -DTF_VERSION=$tf_version

num_cores=$(($(nproc)-3))
cmake --build $build_dir --parallel $num_cores

if [[ "$(uname -m)" == "x86_64" ]]; then
    mv $build_dir/mxrec/core/host/binding.cpython-37m-x86_64-linux-gnu.so $build_dir/mxrec/core/host/binding.so
elif [[ "$(uname -m)" == "aarch64" ]]; then
    mv $build_dir/mxrec/core/host/binding.cpython-37m-aarch64-linux-gnu.so $build_dir/mxrec/core/host/binding.so
else
    echo "Unsupported architecture: $(uname -m)."
    exit 1
fi

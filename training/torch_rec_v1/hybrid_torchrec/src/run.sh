#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
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

rm -rf build
mkdir build
cd build
export CMAKE_BUILD_PARALLEL_LEVEL=24
torch_path=$(python3 -m pip show torch 2>/dev/null | grep -E "^Location:" | awk '{print $2}')/torch/share/cmake
echo "======== current torch_path is: ${torch_path} ========="
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${torch_path}" ..
cmake  --build . --config Release

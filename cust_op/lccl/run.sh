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


msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
export PATH=$parent_dir:$PATH

rm -rf ./custom_op
msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b1 -lan cpp -out ./custom_op -m 0 -op LcclAllToAll
msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b1 -lan cpp -out ./custom_op -m 1 -op LcclAllUss
msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b1 -lan cpp -out ./custom_op -m 1 -op LcclGatherAll

cp -rf op_kernel custom_op/
cp -rf op_host custom_op/

cd custom_op


if [ ! -f "CMakePresets.json" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi


sed -i 's/--nomd5/--nomd5 --nocrc/g' ./cmake/makeself.cmake


sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest":g' CMakePresets.json

cd cmake


if [ ! -f "config.cmake" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi


sed -i 's:"customize":"lccl":g' CMakePresets.json

cd ..

bash build.sh


bash ./build_out/custom_opp*.run

cd ..

rm -rf ./custom_op

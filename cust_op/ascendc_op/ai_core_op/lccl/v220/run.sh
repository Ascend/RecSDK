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

msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
export PATH=$parent_dir:$PATH

ai_core="ai_core-ascend910b1"
if [ "$#" -eq 1 ]; then
  ai_core="$1"
  if [ "$1" = "ai_core-Ascend910B1" ] || 
    [ "$1" = "ai_core-Ascend910_93" ] || 
    [ "$1" = "ai_core-Ascend310P3" ]; then
    echo "ai_core $1"
  else
    echo "ai-core $1 must be one of [ai_core-Ascend910B1 ai_core-Ascend910_93 ai_core-Ascend310P3]"
    exit 1  
  fi
fi

rm -rf ./lccl
msopgen gen -i emb_custom.json -f tf -c ${ai_core} -lan cpp -out ./lccl -m 0 -op LcclAllToAll
msopgen gen -i emb_custom.json -f tf -c ${ai_core} -lan cpp -out ./lccl -m 1 -op LcclAllUss
msopgen gen -i emb_custom.json -f tf -c ${ai_core} -lan cpp -out ./lccl -m 1 -op LcclGatherAll

cp -rf op_kernel lccl/
cp -rf op_host lccl/

cd lccl

if [ ! -f "CMakePresets.json" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi

sed -i 's/--nomd5/--nomd5 --nocrc/g' ./cmake/makeself.cmake
sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest":g' CMakePresets.json
sed -i 's:"customize":"lccl":g' CMakePresets.json

if [ -f "op_host/CMakeLists.txt" ]; then
    sed -i "1 i include(../../../../cmake/func.cmake)" ./op_host/CMakeLists.txt

    line1=$(awk '/tartet_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)/{print NR}' ./op_host/CMakeLists.txt)
    sed -i "${line1}s/OP_TILING_LIB/OP_TILING_LIB LOG_CPP/g" ./op_host/CMakeLists.txt
    
    line2=$(awk '/tartet_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)/{print NR}' ./op_host/CMakeLists.txt)
    sed -i "${line2}s/OP_PROTO_LIB/OP_PROTO_LIB LOG_CPP/g" ./op_host/CMakeLists.txt
fi

bash build.sh


bash ./build_out/custom_opp*.run

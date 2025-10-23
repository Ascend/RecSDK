#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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


VALID_AI_CORES=(
    "ai_core-Ascend910B1"
    "ai_core-Ascend910B2"
    "ai_core-Ascend910B3"
    "ai_core-Ascend910B4"
    "ai_core-Ascend910_93"
    "ai_core-Ascend310P3"
)

validate_ai_core() {
    local input_core="$1"
    for valid_core in "${VALID_AI_CORES[@]}"; do
        if [ "$input_core" = "$valid_core" ]; then
            echo "ai_core $input_core"
            return 0
        fi
    done
    echo "ai core must in : [${VALID_AI_CORES[*]}]" >&2
    exit 1
    return 1
}

ai_core="ai_core-Ascend910B1"
if [ "$#" -eq 1 ]; then
  ai_core="$1"
  validate_ai_core $ai_core
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

# 获取系统ID
os_id=$(cat /etc/os-release | sed -n 's/^ID=//p' | sed 's/^"//;s/"$//')
if [ -z "${os_id}" ]; then
    echo "ERROR: get os_id failed"
    exit 1
fi

# 获取架构
arch=$(uname -m)
if [ -z "${arch}" ]; then
    echo "ERROR: get arch failed"
    exit 1
fi

# 安装编译成功的算子包
bash ./build_out/custom_opp_${os_id}_${arch}.run

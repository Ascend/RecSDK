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
source /etc/profile

# 查找msopgen的路径，加入到环境变量PATH中
msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
export PATH=$parent_dir:$PATH


VALID_AI_CORES=(
    "ai_core-Ascend910"
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

# 利用msopgen生成可编译文件
rm -rf ./cust_op_by_addr
msopgen gen -i emb_custom.json -f tf -c ${ai_core} -lan cpp -out ./cust_op_by_addr -m 0 -op EmbeddingLookupByAddress
msopgen gen -i emb_custom.json -f tf -c ${ai_core} -lan cpp -out ./cust_op_by_addr -m 1 -op EmbeddingUpdateByAddress

cp -rf op_kernel cust_op_by_addr/
cp -rf op_host cust_op_by_addr/

cd cust_op_by_addr

# 判断当前目录下是否存在CMakePresets.json文件
if [ ! -f "CMakePresets.json" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi

# 禁止生成CRC校验和
sed -i 's/--nomd5/--nomd5 --nocrc/g' ./cmake/makeself.cmake

# 修改cann安装路径
sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest":g' CMakePresets.json

cd cmake

# 判断当前目录下是否存在config.cmake文件
if [ ! -f "config.cmake" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi

# 修改设备环境
sed -i 's:set(ASCEND_COMPUTE_UNIT ascend910b):set(ASCEND_COMPUTE_UNIT ascend910b ascend910):g' config.cmake

cd ..

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

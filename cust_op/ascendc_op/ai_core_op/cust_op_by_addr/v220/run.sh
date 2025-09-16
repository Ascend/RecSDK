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
source /etc/profile

# 查找msopgen的路径，加入到环境变量PATH中
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

# 安装编译成功的算子包
bash ./build_out/custom_opp*.run

#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
# Description: Build for cust_op by address
# Author: MindX SDK
# Create: 2023
# History: NA
set -e
source /etc/profile

# 查找msopgen的路径，加入到环境变量PATH中
msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
export PATH=$parent_dir:$PATH

# 利用msopgen生成可编译文件
rm -rf ./custom_op
msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b1 -lan cpp -out ./custom_op -m 0 -op EmbeddingLookupByAddress
msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b1 -lan cpp -out ./custom_op -m 1 -op EmbeddingUpdateByAddress

cp -rf op_kernel custom_op/
cp -rf op_host custom_op/

cd custom_op

# 判断当前目录下是否存在CMakePresets.json文件
if [ ! -f "CMakePresets.json" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi

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


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
# 规避新版本的CANN漏合代码导致GE无法识别到自定义算子问题
sed -i '37i\set(ASCEND_FRAMEWORK_TYPE {plugin})' /usr/local/Ascend/ascend-toolkit/latest/tools/op_project_templates/ascendc/customize/cmake/config.cmake

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

cd ..

rm -rf ./custom_op

#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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


# 查找msopgen的路径，加入到环境变量PATH中
msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
export PATH=$parent_dir:$PATH

ai_core="ai_core-Ascend910B1"
if [ "$#" -eq 1 ]; then
    ai_core=$1
fi

# 利用msopgen生成可编译文件
rm -rf ./asynchronous_complete_cumsum
python3 /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen -i asynchronous_complete_cumsum.json -f tf -c ${ai_core} -lan cpp -out ./asynchronous_complete_cumsum -m 0 -op AsynchronousCompleteCumsum
rm -rf asynchronous_complete_cumsum/op_kernel/*.h
rm -rf asynchronous_complete_cumsum/op_kernel/*.cpp
rm -rf asynchronous_complete_cumsum/host/*.h
rm -rf asynchronous_complete_cumsum/host/*.cpp
cp -rf op_kernel asynchronous_complete_cumsum/
cp -rf op_host asynchronous_complete_cumsum/
cd asynchronous_complete_cumsum

# 判断当前目录下是否存在CMakePresets.json文件
if [ ! -f "CMakePresets.json" ]; then
  echo "ERROR, CMakePresets.json file not exist."
  exit 1
fi

# 禁止生成CRC校验和
sed -i 's/--nomd5/--nomd5 --nocrc/g' ./cmake/makeself.cmake

# 修改cann安装路径
sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest":g' CMakePresets.json
# 修改vendor_name 防止覆盖之前vendor_name为customize的算子;
# vendor_name需要和aclnn中的CMakeLists.txt中的CUST_PKG_PATH值同步，不同步aclnn会调用失败;
# vendor_name字段值不能包含customize；包含会导致多算子部署场景CANN的vendors路径下config.ini文件内容截取错误
sed -i 's:"customize":"asynchronous_complete_cumsum":g' CMakePresets.json

if [ "$ai_core" = "ai_core-Ascend310P3" ]; then
    sed -i "1i #define SUPPORT_V200" ./op_kernel/asynchronous_complete_cumsum.cpp
fi

line=`awk '/ENABLE_SOURCE_PACKAGE/{print NR}' CMakePresets.json`
line=`expr ${line} + 2`
sed -i "${line}s/True/False/g" CMakePresets.json

bash build.sh

# # 安装编译成功的算子包
bash ./build_out/custom_opp*.run

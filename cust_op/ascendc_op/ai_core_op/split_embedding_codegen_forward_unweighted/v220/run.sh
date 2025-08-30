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
    ai_core="$1"
fi

# 利用msopgen生成可编译文件
rm -rf ./split_embedding_codegen_forward_unweighted
python3 /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen -i split_embedding_codegen_forward_unweighted.json -f tf -c ${ai_core} -lan cpp -out ./split_embedding_codegen_forward_unweighted -m 0 -op SplitEmbeddingCodegenForwardUnweighted
rm -rf split_embedding_codegen_forward_unweighted/op_kernel/*.h
rm -rf split_embedding_codegen_forward_unweighted/op_kernel/*.cpp
rm -rf split_embedding_codegen_forward_unweighted/host/*.h
rm -rf split_embedding_codegen_forward_unweighted/host/*.cpp
cp -rf op_kernel split_embedding_codegen_forward_unweighted/
cp -rf op_host split_embedding_codegen_forward_unweighted/

cd split_embedding_codegen_forward_unweighted

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
sed -i 's:"customize":"split_embedding_codegen_forward_unweighted":g' CMakePresets.json

line=`awk '/ENABLE_SOURCE_PACKAGE/{print NR}' CMakePresets.json`
line=`expr ${line} + 2`
sed -i "${line}s/True/False/g" CMakePresets.json

# 增加LOG_CPP编译选项支持错误日志打印
sed -i "1 i include(../../../../cmake/func.cmake)" ./op_host/CMakeLists.txt

line1=`awk '/tartet_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)/{print NR}' ./op_host/CMakeLists.txt`
sed -i "${line1}s/OP_TILING_LIB/OP_TILING_LIB LOG_CPP/g" ./op_host/CMakeLists.txt

line2=`awk '/tartet_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)/{print NR}' ./op_host/CMakeLists.txt`
sed -i "${line2}s/OP_PROTO_LIB/OP_PROTO_LIB LOG_CPP/g" ./op_host/CMakeLists.txt

bash build.sh

# # 安装编译成功的算子包
bash ./build_out/custom_opp*.run
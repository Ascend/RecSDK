#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
rm -rf ./relative_attn_bias_pos
python3 /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen -i relative_attn_bias_pos.json -f tf -c ${ai_core} -lan cpp -out ./relative_attn_bias_pos -m 0 -op RelativeAttnBiasPos
rm -rf relative_attn_bias_pos/op_kernel/*.h
rm -rf relative_attn_bias_pos/op_kernel/*.cpp
rm -rf relative_attn_bias_pos/host/*.h
rm -rf relative_attn_bias_pos/host/*.cpp
cp -rf op_kernel relative_attn_bias_pos/
cp -rf op_host relative_attn_bias_pos/

cd relative_attn_bias_pos

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
sed -i 's:"customize":"relative_attn_bias_pos":g' CMakePresets.json

if [ "$ai_core" = "ai_core-Ascend310P3" ]; then
    sed -i "1i #define SUPPORT_V200" ./op_kernel/relative_attn_bias_pos.h
fi

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

# 安装编译成功的算子包
bash ./build_out/custom_opp*.run

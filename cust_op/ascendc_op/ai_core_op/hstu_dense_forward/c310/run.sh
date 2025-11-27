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
msopgen_path=$(find /usr/local/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
onnx_path=$(dirname "$(readlink -f "$0")")/../../../build/scripts/onnx_plugin

export PATH=$parent_dir:$PATH

VALID_AI_CORES=(
    "ai_core-Ascend910_95"
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
}

ai_core="ai_core-Ascend910_95"
if [ "$#" -eq 1 ]; then
  ai_core="$1"
  validate_ai_core $ai_core
fi

# 利用msopgen生成可编译文件
rm -rf ./hstu_dense_forward
python3 $msopgen_path gen -i ../v220/hstu_dense_forward.json -f tf -c ${ai_core} -lan cpp -out ./hstu_dense_forward -m 0 -op HstuDenseForward
cp -rf ../v220/op_kernel hstu_dense_forward/
cp -rf ../v220/op_host/*.h hstu_dense_forward/op_host/
cp -rf ../v220/op_host/hstu_*.cpp hstu_dense_forward/op_host/
cp -rf ../v220/op_host/tiling_policy.cpp hstu_dense_forward/op_host/
cp -rf ../v220/op_host/tiling_policy_factory.cpp hstu_dense_forward/op_host/
cp -rf ../v220/op_host/tiling_policy_normal.cpp hstu_dense_forward/op_host/
cp -rf ../v220/op_host/tiling_policy_jagged.cpp hstu_dense_forward/op_host/
cp -rf ../v220/op_host/tiling_policy_paged.cpp hstu_dense_forward/op_host/

cd hstu_dense_forward

# 判断当前目录下是否存在CMakePresets.json文件
if [ ! -f "CMakePresets.json" ]; then
  echo "ERROR, CMakePresets.json file not exist."
  exit 1
fi

# 禁止生成CRC校验和
sed -i 's/--nomd5/--nomd5 --nocrc/g' ./cmake/makeself.cmake

# 修改cann安装路径
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest":g' CMakePresets.json
fi
# 修改vendor_name 防止覆盖之前vendor_name为customize的算子;
# vendor_name需要和aclnn中的CMakeLists.txt中的CUST_PKG_PATH值同步，不同步aclnn会调用失败;
# vendor_name字段值不能包含customize；包含会导致多算子部署场景CANN的vendors路径下config.ini文件内容截取错误
sed -i 's:"customize":"hstu_dense_forward":g' CMakePresets.json

line=`awk '/ENABLE_SOURCE_PACKAGE/{print NR}' CMakePresets.json`
line=`expr ${line} + 2`
sed -i "${line}s/True/False/g" CMakePresets.json

sed -i "1i #define SUPPORT_910_95" ./op_kernel/matmul_constexpr.h

add_cmake_line="install(FILES \${CMAKE_CURRENT_SOURCE_DIR}/../../../v220/hstu_dense_forward.json DESTINATION packages/vendors/\${vendor_name}/op_impl/ai_core/tbe/\${vendor_name}_impl/dynamic)"
sed -i '$a\'"$add_cmake_line" ./op_kernel/CMakeLists.txt
sed -i '1i\add_ops_compile_options(ALL OPTIONS --cce-long-call=true)' ./op_kernel/CMakeLists.txt

# 增加LOG_CPP编译选项支持错误日志打印
sed -i "1 i include(../../../../cmake/func.cmake)" ./op_host/CMakeLists.txt

line1=`awk '/target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)/{print NR}' ./op_host/CMakeLists.txt`
sed -i "${line1}s/OP_TILING_LIB/OP_TILING_LIB LOG_CPP/g" ./op_host/CMakeLists.txt

line2=`awk '/target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)/{print NR}' ./op_host/CMakeLists.txt`
sed -i "${line2}s/OP_PROTO_LIB/OP_PROTO_LIB LOG_CPP/g" ./op_host/CMakeLists.txt

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
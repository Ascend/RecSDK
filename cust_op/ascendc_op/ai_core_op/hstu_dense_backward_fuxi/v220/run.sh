#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

# 利用msopgen生成可编译文件
rm -rf ./hstu_dense_backward_fuxi
python3 /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen -i hstu_dense_backward.json -f tf -c ${ai_core} -lan cpp -out ./hstu_dense_backward_fuxi -m 0 -op HstuDenseBackwardFuxi
rm -rf hstu_dense_backward_fuxi/op_kernel/*.h
rm -rf hstu_dense_backward_fuxi/op_kernel/*.cpp
rm -rf hstu_dense_backward_fuxi/host/*.h
rm -rf hstu_dense_backward_fuxi/host/*.cpp
cp -rf op_kernel hstu_dense_backward_fuxi/
cp -rf op_host hstu_dense_backward_fuxi/

cd hstu_dense_backward_fuxi

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
sed -i 's:"customize":"hstu_dense_backward_fuxi":g' CMakePresets.json

line=`awk '/ENABLE_SOURCE_PACKAGE/{print NR}' CMakePresets.json`
line=`expr ${line} + 2`
sed -i "${line}s/True/False/g" CMakePresets.json

add_cmake_line="install(FILES \${CMAKE_CURRENT_SOURCE_DIR}/../../hstu_dense_backward.json DESTINATION packages/vendors/\${vendor_name}/op_impl/ai_core/tbe/\${vendor_name}_impl/dynamic)"
sed -i '$a\'"$add_cmake_line" ./op_kernel/CMakeLists.txt

# 增加LOG_CPP编译选项支持错误日志打印
sed -i "1 i include(../../../../cmake/func.cmake)" ./op_host/CMakeLists.txt

line1=`awk '/tartet_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)/{print NR}' ./op_host/CMakeLists.txt`
sed -i "${line1}s/OP_TILING_LIB/OP_TILING_LIB LOG_CPP/g" ./op_host/CMakeLists.txt

line2=`awk '/tartet_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)/{print NR}' ./op_host/CMakeLists.txt`
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

# 只允许字母/数字/点/下划线/连字符（覆盖常见 os_id 与 arch）
SAFE_REGEX='^[A-Za-z0-9._-]+$'
if ! [[ "$os_id" =~ $SAFE_REGEX ]]; then
    echo "ERROR: invalid os_id: $os_id" >&2
    exit 1
fi
if ! [[ "$arch" =~ $SAFE_REGEX ]]; then
    echo "ERROR: invalid arch: $arch" >&2
    exit 1
fi

# 安装编译成功的算子包
installer="./build_out/custom_opp_${os_id}_${arch}.run"
bash -- "$installer"
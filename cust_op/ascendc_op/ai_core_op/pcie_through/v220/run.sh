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
}

ai_core="ai_core-Ascend910B1"
if [ "$#" -eq 1 ]; then
  ai_core="$1"
  validate_ai_core $ai_core
fi

rm -rf ./pcie_through
if [ "${ai_core}" == "ai_core-Ascend910B1" ]; then
  msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b1 -lan cpp -out ./pcie_through -m 0 -op RmaSwapMultiTables
elif [ "${ai_core}" == "ai_core-Ascend910_93" ]; then
  msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910_93 -lan cpp -out ./pcie_through -m 0 -op RmaSwapMultiTables
else
  echo "Unsupported chip type ${ai_core}"
fi

cp -rf op_kernel pcie_through/
cp -rf op_host pcie_through/

cd pcie_through

if [ ! -f "CMakePresets.json" ]; then
  echo "CMakePresets.json does not exist in current directory"
  exit 1
fi

sed -i 's/--nomd5/--nomd5 --nocrc/g' ./cmake/makeself.cmake

if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest":g' CMakePresets.json
fi
cd cmake

if [ ! -f "config.cmake" ]; then
  echo "config.cmake does not exist in current directory"
  exit 1
fi

if [ "${ai_core}" == "ai_core-Ascend910B1" ]; then
  sed -i 's:set(ASCEND_COMPUTE_UNIT ascend910b):set(ASCEND_COMPUTE_UNIT ascend910b ascend910):g' config.cmake
elif [ "${ai_core}" == "ai_core-ascend910_93" ]; then
  sed -i 's:set(ASCEND_COMPUTE_UNIT ascend910_93):set(ASCEND_COMPUTE_UNIT ascend910_93 ascend910):g' config.cmake
else
  echo "Unsupported chip type ${ai_core}"
fi

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

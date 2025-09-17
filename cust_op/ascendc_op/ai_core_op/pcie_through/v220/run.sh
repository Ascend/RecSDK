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

msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
export PATH=$parent_dir:$PATH

ai_core="ai_core-Ascend910B1"
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

rm -rf ./pcie_through
if [ "${ai_core}" == "ai_core-Ascend910B1" ]; then
  msopgen gen -i emb_custom.json -f tf -c ai_core-Ascend910B1 -lan cpp -out ./pcie_through -m 0 -op RmaSwapMultiTables
elif [ "${ai_core}" == "ai_core-ascend910_93" ]; then
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

sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest":g' CMakePresets.json

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

bash ./build_out/custom_opp*.run

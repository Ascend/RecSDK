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

CHIP_TYPE="910B"  # support：910B 910_93

msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
export PATH=$parent_dir:$PATH

rm -rf ./custom_op
if [ "$CHIP_TYPE" == "910B" ]; then
  msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b1 -lan cpp -out ./custom_op -m 0 -op RmaSwapMultiTables
elif [ "$CHIP_TYPE" == "910_93" ]; then
  msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910_93 -lan cpp -out ./custom_op -m 0 -op RmaSwapMultiTables
else
  echo "Unsupported chip type $CHIP_TYPE"
fi

cp -rf op_kernel custom_op/
cp -rf op_host custom_op/

cd custom_op

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

if [ "$CHIP_TYPE" == "910B" ]; then
  sed -i 's:set(ASCEND_COMPUTE_UNIT ascend910b):set(ASCEND_COMPUTE_UNIT ascend910b ascend910):g' config.cmake
elif [ "$CHIP_TYPE" == "910_93" ]; then
  sed -i 's:set(ASCEND_COMPUTE_UNIT ascend910_93):set(ASCEND_COMPUTE_UNIT ascend910_93 ascend910):g' config.cmake
else
  echo "Unsupported chip type $CHIP_TYPE"
fi

cd ..

bash build.sh

bash ./build_out/custom_opp*.run

cd ..

rm -rf ./custom_op
#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

##################################################################
#   run_presmoke.sh 预冒烟验证
##################################################################

set -e
echo "================        run presmoke test!!!!        ================"
export PROJECT_DIR=$(realpath $(pwd)/../)
export VENDORS=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/
export PRESMOKE_DIR=$PROJECT_DIR/build/presmoke/
export PTA_DIR=$PROJECT_DIR/cust_op/framework/torch_plugin/torch_library/common/
export WORLD_SIZE=2
export ASCEND_RT_VISIBLE_DEVICES=0,1

source /opt/buildtools/torch_v2_pt2.7.1/bin/activate
source /usr/local/set_cann_env.sh a2

echo "----------------        install torch_npu        ----------------"
pip3 list | grep -E "(torch|fbgemm)"

echo "----------------        match cases        ----------------"
cd $PRESMOKE_DIR
rm -fr changes.txt
if [[ -f /workspace/change.txt ]]; then
    cp -f /workspace/change.txt ./changes.txt
else
    echo "/workspace/change.txt not found."
    git fetch origin && git diff --name-only HEAD..origin/develop > changes.txt
fi
cat changes.txt
python3 control.py

echo "================        run presmoke success        ================"

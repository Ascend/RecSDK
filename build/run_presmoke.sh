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
export PROJECT_DIR=$(pwd)/../

export VENDORS=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/

export PRESMOKE_DIR=$PROJECT_DIR/build/presmoke/

export PTA_DIR=$PROJECT_DIR/cust_op/framework/torch_plugin/torch_library/common/

source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib/:${LD_LIBRARY_PATH}
export PATH=/usr/local/python3.11.0/bin:${PATH}

unset ASCEND_CUSTOM_OPP_PATH

echo "----------------        install torch_npu        ----------------"
cd $PRESMOKE_DIR
obsutil cp obs://mindcluster/torch_npu-2.7.1.post3-cp311-cp311-manylinux_2_28_aarch64.whl ./ -f -r
python3 -m pip install torch_npu-2.7.1.post3-cp311-cp311-manylinux_2_28_aarch64.whl

echo "----------------        build pta        ----------------"
cd $PTA_DIR && dos2unix build_ops.sh && bash build_ops.sh

echo "----------------        match cases        ----------------"
cd $PRESMOKE_DIR
git fetch origin
git diff --name-only HEAD..origin/develop > changes.txt
cat changes.txt

python3 control.py

echo "================        run presmoke success        ================"


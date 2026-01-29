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
#   run_presmoke_tf.sh TF框架预冒烟验证
##################################################################

set -e
#################### 工具函数 ####################
log()  { echo "$(date '+%F %T') [INFO] $*" ; }
warn() { echo >&2 -e "\033[1;31m[$(date '+%F %T')][WARN] $*\033[0m" ; }
die()  { echo >&2 -e "\033[1;31m[$(date '+%F %T')][FAIL] $*\033[0m"; exit 1; }

#################### 入参校验 ####################
[[ $# -eq 1 ]] || die "Usage: $0 <workspace_dir>"
WORKSPACE=$(realpath "$1")
[[ -d $WORKSPACE ]] || die "Workspace $WORKSPACE not exists or not a directory"

#################### 全局变量 ####################
CODE_DIR=${WORKSPACE}/RecSDK  # develop分支源码存放路径
MODEL_DIR=${WORKSPACE}/RecSDK_examples  # develop_examples_and_tools分支源码存放路径
ARCH=$(uname -m)
PKG_NAME=Ascend-mindxsdk-mxrec*${ARCH}.tar.gz
PKG_PATH=${WORKSPACE}/${PKG_NAME}
TF_WHL=mindxsdk-mxrec/tf1_whl/mx_rec-*.whl
DEMO_DIR="little_demo"
INSTALL_BASE="${WORKSPACE}/workspace_tf"

#################### 清理旧环境 ####################
rm -rf ${INSTALL_BASE}
mkdir -p ${INSTALL_BASE}
cd ${INSTALL_BASE} || die "cd ${INSTALL_BASE} failed"

#################### rec 包安装 ####################
log "Extracting ${PKG_NAME} ..."
cp -rf ${PKG_PATH} ./ || die "cp pkg failed"
tar -zxvf ${PKG_NAME} || die "tar failed"
log "Installing mx_rec wheel ..."
pip3 install ${TF_WHL} --force-reinstall --no-cache-dir || die "pip install failed"

#################### 拷贝示例 ####################
[[ -d ${MODEL_DIR}/examples/demo/${DEMO_DIR} ]] || die "Demo directory not found"
cp -rf ${MODEL_DIR}/examples/demo/${DEMO_DIR} ./ || die "cp demo failed"
cd ${DEMO_DIR} || die "cd demo failed"

#################### 网络IP获取 ####################
IP=$(hostname -I | awk '{print $1}')
[[ -n $IP ]] || die "Cannot obtain local IP"
log "Detected IP: $IP"

#################### 运行示例 ####################
log "Running demo ..."
rm -rf temp*.log
source /usr/local/Ascend/ascend-toolkit/set_env.sh
sed -i -e 's/local_rank_size=8/local_rank_size=1/g' run.sh
bash run.sh main.py $IP
COUNT=$(grep "Demo done" temp*.log | wc -l)
if [[ $COUNT -eq 1 ]]; then
    log "----------------  presmoke_tf executed successfully!  ----------------"
else
    die "Case failed"
fi
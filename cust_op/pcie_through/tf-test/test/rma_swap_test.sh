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

rm -rf /root/ascend/log

clearShm(){ # 清理残留的共享内存
  ids=$(ipcs -m | awk '/^0x/ {print $2}')
  for id in $ids; do
    echo "sudo ipcrm -m ${id}"
    ipcrm -m "$id"
  done
}

local_rank_size=1 # 每个节点使用的NPU卡数

export LD_PRELOAD=/lib64/libgomp.so.1
export RMA_DEVICE_ID=0  #指定device

current_dir=$(pwd)
echo "当前目录是：$current_dir"

# 获取上一级目录
parent_dir=$(dirname "${current_dir}")
export PYTHONPATH=${current_dir}/../src/build/pybind/:${PYTHONPATH}
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize

DATE=$(date +%Y-%m-%d)
DATE2=$(date +%H-%M-%S)

export run_py=rma_swap_multi_tables.py
export ai_type=aicore    #aicpu aicore tdt

# 使用py脚本名称作为存放log的文件目录
logdir=${ai_type}_${run_py%%.*}_${DATE}
if test -d "$logdir"; then
    echo "文件夹存在"
else
    mkdir ${logdir}
fi

shapelist=(64 128 256 512 1024 2048 4096 8192)
step=1000
table_num=1
# 遍历list

echo "=========${run_py}============"
for shape in "${shapelist[@]}"; do
  echo "----shape: ${shape}----"
  clearShm
  mpirun -np ${local_rank_size} --allow-run-as-root python3 ${run_py} --shape ${shape} --step ${step} --table_num ${table_num} \
  2>&1 | tee "${logdir}/${run_py}_${shape}_${step}_${table_num}.log"
  wait $!

  grep "bandwidth" "${logdir}/${run_py}_${shape}_${step}_${table_num}.log"
  wait $!
  clearShm
done
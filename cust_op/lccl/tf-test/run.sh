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
local_rank_size=$1
py=$2

rm -rf /root/atc_data/
rm -rf /root/ascend/*
rm -rf kernel_meta_*
ulimit -c 0

mpi_path=/usr/local/openmpi/bin/
mx_rec_package_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/mx_rec
so_path=${mx_rec_package_path}/libasc
tfa_path=/usr/local/Ascend/tfplugin/latest/python/site-packages/npu_bridge  # tf1
# tfa_path=/usr/local/Ascend/tfplugin/latest/python/site-packages/npu_device/compat/v1  # tf2
export LD_LIBRARY_PATH=${so_path}:${tfa_path}:/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${so_path}:$PYTHONPATH
export PATH=${mpi_path}/bin:$PATH
export LD_PRELOAD=/lib64/libgomp.so.1

export ASCEND_GLOBAL_LOG_LEVEL=0 # “设置日志级别”章节0:debug, 1:info, 2:warning, 3:error, 4:NULL
export MXREC_LOG_LEVEL=DEBUG
export TF_CPP_MIN_LOG_LEVEL=3

export HCCL_BUFFSIZE=1
export BETTER_EXCEPTIONS=1

for i in $(ipcs -m | tail -n +4 | awk {'print $2'}); do
    ipcrm -m $i
done

host=localhost
num_process=${local_rank_size}
host_string=${host//_/:${local_rank_size},node}:${local_rank_size}
echo run in $host_string

interface="lo"
mpi_args='-x BIND_INFO="0:48 48:48 96:48" -x MXREC_LOG_LEVEL=DEBUG -bind-to none'
env
horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_rags}" --mpi -H localhost:${local_rank_size} \
    python3.7 ${py} --local_rank_size ${local_rank_size} --hccl_json hccl_json_${local_rank_size}p.json | tee temp.log

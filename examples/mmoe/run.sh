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

cur_path=$(dirname "$(readlink -f "$0")")

so_path=$1
rec_package_path=$2
hccl_cfg_json=$3
dlrm_criteo_data_path=$4
ip=$5  # no ranktable时传入该参数

interface="lo"
num_server=1
local_rank_size=8
num_process=$((num_server * local_rank_size))
export TRAIN_RANK_SIZE=$num_process

################# 参数配置 ######################
export USE_DYNAMIC=0            # 0：静态shape；1：动态shape
export CACHE_MODE="HBM"         # HBM；DDR；SSD
export USE_FAAE=0               # 0：关闭准入淘汰；1：开启准入淘汰
export USE_DYNAMIC_EXPANSION=0  # 0：关闭动态扩容；1: 开启动态扩容
export USE_MULTI_LOOKUP=0       # 0：一表一查；1：一表多查
export USE_MODIFY_GRAPH=0       # 0：feature spec模式；1：自动改图模式
export USE_DP=0                 # 0：关闭DP；1：开启DP
################################################
echo "CACHE_MODE:${CACHE_MODE}"

export HCCL_CONNECT_TIMEOUT=1200
export DLRM_CRITEO_DATA_PATH=${dlrm_criteo_data_path}
export PYTHONPATH=${rec_package_path}:${so_path}:$PYTHONPATH
export LD_PRELOAD=/usr/lib64/libgomp.so.1:/usr/lib64/libstdc++.so.6
export LD_LIBRARY_PATH=${so_path}:/usr/local/lib:$LD_LIBRARY_PATH
export ASCEND_DEVICE_ID=0
export RANK_ID_START=0
export JOB_ID=10086
export CUSTOMIZED_OPS_LIB_PATH=${so_path}/libcust_ops.so # Todo: please config
export MXREC_LOG_LEVEL="INFO"
export TF_CPP_MIN_LOG_LEVEL=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ENABLE_FORCE_V2_CONTROL=1

export PROFILING_OPTIONS='{"output":"/home/yz/profiling",
                           "training_trace":"on",
                           "task_trace":"on",
                           "aicpu":"on",
                           "fp_point":"",
                           "bp_point":"",
                           "aic_metrics":"PipeUtilization"}'

RANK_ID_START=0

export MXREC_MODE="ASC"
echo "MXREC_MODE is $MXREC_MODE"
export py=main_mxrec.py
echo "py is $py"

# 区分ranktable和no ranktable
if [ -n "$ip" ]; then
    # no ranktable分支
    echo "Current is no ranktable solution."
    echo "Input node ip: $ip, please make sure this ip is available."
    export CM_CHIEF_IP=$ip  # 主节点ip
    export CM_CHIEF_PORT=60001  # 主节点监听端口
    export CM_CHIEF_DEVICE=0  # 主节点device id
    export CM_WORKER_IP=$ip  # 当前节点ip
    export CM_WORKER_SIZE=$num_process  # 参与集群训练的device数量
    echo "CM_CHIEF_IP=$CM_CHIEF_IP"
    echo "CM_CHIEF_PORT=$CM_CHIEF_PORT"
    echo "CM_CHIEF_DEVICE=$CM_CHIEF_DEVICE"
    echo "CM_WORKER_IP=$CM_WORKER_IP"
    echo "CM_WORKER_SIZE=$CM_WORKER_SIZE"
else
    # ranktable分支
    echo "Current is ranktable solution, hccl json file:${hccl_cfg_json}"
    export RANK_SIZE=$num_process
    echo "RANK_SIZE=${RANK_SIZE}, please make sure hccl configuration json file match this parameter"
    export RANK_TABLE_FILE=${hccl_cfg_json}
fi

echo "use horovod to start tasks"
# GLOG_stderrthreshold -2:TRACE -1:DEBUG 0:INFO 1:WARN 2.ERROR, 默认为INFO
mpi_args='-x BIND_INFO="0:12 12:48 60:48" -x GLOG_stderrthreshold=2 -x GLOG_logtostderr=true -bind-to none -x NCCL_SOCKET_IFNAME=docker0 -mca btl_tcp_if_exclude docker0'

horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_args}" --mpi -H localhost:${local_rank_size} \
python3.7 ${py} 2>&1 | tee temp_${CACHE_MODE}_${num_process}p.log

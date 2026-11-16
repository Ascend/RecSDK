#!/bin/bash
# Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.
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
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_BUFFSIZE=128
export MULTI_STREAM_MEMORY_RUSE=2

RECSYS_DIR=$(realpath ../)
HSTU_DIR=$RECSYS_DIR/hstu
# 根据实际情况设置python引用路径
MEGATRON_DIR=$RECSYS_DIR/../../Megatron-LM/
MINDSPEED_DIR=$RECSYS_DIR/../../MindSpeed/
export PYTHONPATH=${PYTHONPATH}:${RECSYS_DIR}:${HSTU_DIR}:${MEGATRON_DIR}:${MINDSPEED_DIR}

#---------------------------------------------
# speedup
#---------------------------------------------
export TASK_QUEUE_ENABLE=2

# 根据实际情况修改
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1  # 需要修改为节点的真实ip
export MASTER_PORT=6000
NPROC_PER_NODE=8
export WORLD_SIZE=$(($NPROC_PER_NODE * $NNODES))
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# cpu-binding
NPU_NUM=${NPROC_PER_NODE}
CPU_CORES=$(nproc --all)
if [ "$NPU_NUM" -eq 0 ]; then
  echo "NPU_NUM is 0, exit"
  exit 1
fi
CORES_PER_NPU=$((CPU_CORES / NPU_NUM))
CPU_AFFINITY_CONF_TMP=1
if [ "$NPU_NUM" -gt 0 ]; then
  for (( i=0; i<NPU_NUM; i++)); do
    start_core=$(( i * CORES_PER_NPU))
    end_core=$((start_core + CORES_PER_NPU -1))
    CPU_AFFINITY_CONF_TMP+=",npu${i}:${start_core}-${end_core}"
  done
fi
export CPU_AFFINITY_CONF=$CPU_AFFINITY_CONF_TMP
echo "CPU_AFFINITY_CONF="$CPU_AFFINITY_CONF


#---------------------------------------------
# prof related
# 默认关闭；可通过 benchmark/configs 中 run_cmd 的 env 前缀或 profiling_flag 字段覆盖
#---------------------------------------------
export MODEL_PROFILING_FLAG=${MODEL_PROFILING_FLAG:-0}

#---------------------------------------------
# hccl related
# HCCL 确定性开关，默认关闭；精度比对时设为 TRUE
#---------------------------------------------
export HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-FALSE}
# profiling 模式下强制开启 HCCL 确定性，保证多次采集结果可比
case "${MODEL_PROFILING_FLAG}" in
    1|true|TRUE|True) export HCCL_DETERMINISTIC=TRUE ;;
esac

#---------------------------------------------
# train job related
#---------------------------------------------
py_file=./training/pretrain_gr_ranking.py
config_file=./training/configs_extra/ranking_gr_7B.gin

export USE_FSDP2=1
MICRO_BATCH_SIZE=12
GLOBAL_BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH_SIZE))
export CUDA_DEVICE_MAX_CONNECTIONS=8
GPT_ARGS="
  --num-layers 9 \
  --num-attention-heads 4 \
  --hidden-size 4096 \
  --seq-length 200 \
  --max-position-embeddings 200 \
  --use-torch-fsdp2 \
  --fsdp2-config-path ${HSTU_DIR}/configs/7B_fsdp2_config.yaml \
  --no-gradient-accumulation-fusion \
  --untie-embeddings-and-output-weights \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  "

torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${py_file} \
    --gin-config-file ${config_file} \
    ${GPT_ARGS} \
    --epochs 50 \
    $@ 2>&1 |tee ranking_gr_7B_$(date '+%Y%m%d_%H%M%S').log

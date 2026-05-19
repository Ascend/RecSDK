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

# cpu-binding
NPU_NUM=$(npu-smi info|grep 950|wc -l)
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
#---------------------------------------------
export NPU_PROFILE=0

#---------------------------------------------
# train job related
#---------------------------------------------
py_file=pretrain_gr_ranking.py
config_file=movielen_ranking.gin

# 根据实际情况修改
export WORLD_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_FSDP2=1
MICRO_BATCH_SIZE=128
GLOBAL_BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH_SIZE))
export CUDA_DEVICE_MAX_CONNECTIONS=2 # 与mindspeed文档保持一致，默认值为2
GPT_ARGS="
  --num-layers 1 \
  --num-attention-heads 4 \
  --hidden-size 128 \
  --seq-length 200 \
  --max-position-embeddings 200 \
  --use-torch-fsdp2 \
  --fsdp2-config-path ${HSTU_DIR}/configs/fsdp2_config.yaml \
  --no-gradient-accumulation-fusion \
  --untie-embeddings-and-output-weights \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  "

torchrun \
    --nproc_per_node ${WORLD_SIZE} \
    --master_addr localhost \
    --master_port 6000 \
    ${py_file} \
    --gin-config-file ${config_file} \
    ${GPT_ARGS} \
    2>&1 |tee temp_$(date '+%Y%m%d_%H%M%S').log

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

RECSYS_DIR=$(realpath ../)
HSTU_DIR=$RECSYS_DIR/hstu
# 根据实际情况设置python引用路径
MEGATRON_DIR=$RECSYS_DIR/../../Megatron-LM/
MINDSPEED_DIR=$RECSYS_DIR/../../MindSpeed/
export PYTHONPATH=${PYTHONPATH}:${RECSYS_DIR}:${HSTU_DIR}:${MEGATRON_DIR}

#---------------------------------------------
# prof related
#---------------------------------------------
export MODEL_PROFILING_FLAG=0

#---------------------------------------------
# train job related
#---------------------------------------------
py_file=./training/pretrain_gr_ranking.py
config_file=./training/configs_extra/ranking_gr_7B.gin

# 根据实际情况修改
export WORLD_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export USE_FSDP2=1
export FSDP2_CONFIG_PATH=${PWD}/configs/7B_gpu_fsdp2_config.yaml
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
  --fsdp2-config-path ${HSTU_DIR}/configs/7B_gpu_fsdp2_config.yaml \
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
    --epochs 50 \
    $@ 2>&1 |tee ranking_gr_7B_$(date '+%Y%m%d_%H%M%S').log

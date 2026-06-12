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
export PYTHONPATH=${PYTHONPATH}:${RECSYS_DIR}:${HSTU_DIR}:${MEGATRON_DIR}

#---------------------------------------------
# prof related
#---------------------------------------------
export GPU_PROFILE=1

#---------------------------------------------
# precision config
#---------------------------------------------
export PRECISION_FLAG=0

#---------------------------------------------
# train job related
#---------------------------------------------
py_file=pretrain_gr_ranking.py
config_file=movielen_ranking.gin

# 根据实际情况修改
WORLD_SIZE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MICRO_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH_SIZE))
export CUDA_DEVICE_MAX_CONNECTIONS=2
export GR_2B=1

torchrun \
    --nproc_per_node ${WORLD_SIZE} \
    --master_addr localhost \
    --master_port 6000 \
    ${py_file} \
    --gin-config-file ${config_file} \
    2>&1 |tee temp_$(date '+%Y%m%d_%H%M%S').log

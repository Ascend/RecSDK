#!/bin/bash
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#----------------------------------------
# lib related
#----------------------------------------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_PRELOAD=/usr/lib64/libgomp.so.1

SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
TORCH_LIB_PATH="$SITE_PACKAGES/torch/lib"
CUSTOM_LIB_PATH="$SITE_PACKAGES/torchrec_embcache"
export LD_LIBRARY_PATH="$SITE_PACKAGES:$TORCH_LIB_PATH:$CUSTOM_LIB_PATH:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=12

# 算子适配层文件libfbgemm_npu_api.so的路径
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"

#----------------------------------------
# ascend related
#----------------------------------------
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export ASCEND_GLOBAL_LOG_LEVEL=3
# export ASCEND_GLOBAL_EVENT_ENABLE=0
# export ASCEND_SLOG_PRINT_TO_STDOUT=1

#----------------------------------------
# embcache related
#----------------------------------------
export WITH_EMBCACHE=1
export INIT_LINEAR=1
export ENABLE_GLOBAL_UNIQUE=0


# 供参考：16GB=17179869184; 30GB=30*1024*1024*1024=32212254720;
export EMBCACHE_SIZE_ON_DEVICE_MEM=$((1*1024*1024))

# ENABLE_FAST_HASHMAP=false时，默认适用unordered_map
export ENABLE_FAST_HASHMAP=false
# 2*1024*1024=2097152
export FAST_HASHMAP_RESERVE_BUCKET_NUM=2097152


#----------------------------------------
# training job related
#----------------------------------------
export WORLD_SIZE=2
export ASCEND_RT_VISIBLE_DEVICES=6,7

pytest ./test_embedding_cache_pipeline.py
pytest ./test_embedding_ec_cache_pipeline.py
pytest ./test_save_and_load.py
pytest ./test_kjt_with_time.py

export ENABLE_GLOBAL_UNIQUE=1  # feature admit is related with global unique
export DO_EC_LOCAL_UNIQUE=1
pytest ./test_feature_filter.py

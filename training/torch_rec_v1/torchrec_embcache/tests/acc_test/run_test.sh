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
if [ -e "/usr/lib64/libgomp.so.1" ]; then
    export LD_PRELOAD=/usr/lib64/libgomp.so.1
fi

SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
TORCH_LIB_PATH="$SITE_PACKAGES/torch/lib"
CUSTOM_LIB_PATH="$SITE_PACKAGES/torchrec_embcache"
export LD_LIBRARY_PATH="$SITE_PACKAGES:$TORCH_LIB_PATH:$CUSTOM_LIB_PATH:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=12

#----------------------------------------
# ascend related
#----------------------------------------
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

#----------------------------------------
# embcache related
#----------------------------------------
export WITH_EMBCACHE=1


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

(
    export ENABLE_PARALLEL_GLOBAL_UNIQUE=1
    pytest ./test_embedding_ec_cache_pipeline.py
)

(
    export DO_EC_LOCAL_UNIQUE=1
    export LOCAL_UNIQUE_PARALLEL_BATCH_NUM=4
    pytest ./test_embedding_ec_cache_pipeline.py
)

(
    export DO_EC_LOCAL_UNIQUE=1
    pytest ./test_feature_filter.py
)

pytest ./test_feature_filter.py
pytest ./test_kjt_with_time.py
pytest ./test_save_and_load.py
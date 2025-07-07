#!/bin/bash

#----------------------------------------
# lib related
#----------------------------------------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_PRELOAD=/usr/lib64/libgomp.so.1

SITE_PACKAGES=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
TORCH_LIB_PATH="$SITE_PACKAGES/torch/lib"
CUSTOM_LIB_PATH="$SITE_PACKAGES/embedding_cache"
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$CUSTOM_LIB_PATH:$LD_LIBRARY_PATH"

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

#!/bin/bash
export USE_NPU_HSTU=1                                                                 # 是否使用hstu算子加速
export ENABLE_RAB=0                                                                   # 是否带RAB
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"                     # 根据实际情况修改
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3                                              # 根据实际情况修改
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345 | tee temp.log

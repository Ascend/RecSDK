#!/bin/bash
echo -e "\n--> Generative Recommenders on ROMA environment\n"

cfg_path=/onetrans/onetrans_rank_kr.ini
train_data_path=/onetrans/data/train_sample
test_data_path=/onetrans/data/test_sample
save_path=/onetrans/save
tensorboard_path=/onetrans/tensorbaord

export check_path=/onetrans/

#compile
export ENABLE_COMPILE=0
export ENABLE_GRAPH=0

#精度校验开关
export Check_Point=0

#设备号设置
VISIBLE_DEVICES=3
export NPU_FLAG=False
export ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}



unset ASCEND_CUSTOM_OPP_PATH
cur_dir=$(dirname "$0")

# Set environment variables
export GLOO_SOCKET_IFNAME='eth0'
export TP_SOCKET_IFNAME='eth0'
export HCCL_SOCKET_IFNAME='eth0'
export CUDA_DEVICE_MAX_CONNECTIONS='1'
export HCCL_WHITELIST_DISABLE='1'
export INF_NAN_MODE_ENABLE='1'
export HCCL_ASYNC_ERROR_HANDLING='0'
export WITHOUT_JIT_COMPILE='1'
# export HCCL_OP_BASE_FFTS_MODE_ENABLE='FALSE'
export COMBINED_ENABLE='1'
export OMP_NUM_THREADS='1'
export LD_LIBRARY_PATH=/usr/local/gcc10.2.0/lib64:/usr/local/Ascend/toolbox/latest/Ascend-DMI/lib64:/usr/lib/aarch64-linux-gnu/hdf5/serial:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:/usr/local/seccomponent/lib/:/usr/local/seccomponent/lib/openssl/:/usr/local/mindspore-lite/mindspore-lite-2.2.12-linux-aarch64/tools/converter/lib:/usr/local/mindspore-lite/mindspore-lite-2.2.12-linux-aarch64/runtime/lib:/usr/local/mindspore-lite/mindspore-lite-2.2.12-linux-aarch64/runtime/third_party/dnnl:/usr/lib64:/home/ma-user/anaconda3/envs/python311/lib
export ASCEND_PROCESS_LOG_PATH=/var/log/npu/slog
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=6000


# log
# export ASCEND_SLOG_PRINT_TO_STDOUT=0   # 日志打屏, 可选
# export ASCEND_GLOBAL_LOG_LEVEL=3       # 日志级别常用 1 INFO级别; 3 ERROR级别
# export ASCEND_GLOBAL_EVENT_ENABLE=0    # 默认不使能event日志信息
# export ASCEND_LAUNCH_BLOCKING=1       # 默认不开启算子下发同步，影响训练性能; 开启后每执行完一个算子会做一次流同步

export RANK=6
export WORLD_SIZE=1
export LOCAL_RANK=6
NONSEQ_LIST=(512)

EVAL_BS_LIST=(256 640)
DMODEL_LIST=(64 256)

for Batch_Size in "${EVAL_BS_LIST[@]}"; do
  for Non_Seq_Len in "${NONSEQ_LIST[@]}"; do
    for Dim in "${DMODEL_LIST[@]}"; do

      seq_len=$((Non_Seq_Len * 2))
      export profiling_dir="onetrans/L20inductor_nograph_BatchSize${Batch_Size}_Seqlen${seq_len}_Dim${Dim}"
      export Batch_Size="${Batch_Size}"
      export Non_Seq_Len="${Non_Seq_Len}"
      export Dim="${Dim}"

      cur_dir=$(dirname "$0")
      echo "========================================"
      echo "Running with bs=${Batch_Size}, non_seq=${Non_Seq_Len}, d_model=${Dim}"
      echo "profiling_dir=${profiling_dir}"
      echo "========================================"

      torchrun --nproc_per_node=1 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=12349 \
        main_rank.py \
        --config_file="$cfg_path" \
        --train_data_dir="$train_data_path" \
        --test_data_dir="$test_data_path" \
        --save_dir="$save_path" \
        --is_train=False \
        --tensorboard_log_dir="$tensorboard_path" \
        --open_megatron=False

    done
  done
done
#!/bin/bash
echo -e "\n--> Generative Recommenders on ROMA environment\n"

#修改下面路径
cfg_path=/onetrans_rank_kr.ini   #修改为onetrans_rank_kr.ini所在的绝对路径
train_data_path=/train_sample    #修改为数据集train_sample所在的绝对路径
test_data_path=/test_sample      #修改为数据集test_sample所在的绝对路径
save_path=/save    #修改为数据模型运行结果所在的绝对路径，为自己创建的/save路径
tensorboard_path=/tensorbaord    #修改为数据模型运行结果所在的绝对路径，为自己创建的/tensorbaord路径

#compile
export ENABLE_COMPILE=0
export ENABLE_GRAPH=0

#精度校验开关
export Check_Point=0
export check_path=/onetrans/    #精度检验时修改为保存.pth的绝对路径

#设备号设置
VISIBLE_DEVICES=3
export NPU_FLAG=False
export ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}

#修改为# rec算子libfbgemm_npu_api.so的路径
export LIB_FBGEMM_NPU_API_SO_PATH=

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
export ASCEND_PROCESS_LOG_PATH=/var/log/npu/slog
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=6000

export RANK=1
export WORLD_SIZE=1
export LOCAL_RANK=${VISIBLE_DEVICES}
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
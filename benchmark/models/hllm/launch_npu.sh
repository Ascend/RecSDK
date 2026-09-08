# 设备号
DEVICE_ID=0
export ASCEND_VISIBLE_DEVICES=${DEVICE_ID}
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_ID}

export DEVICE_TYPE="npu"

export MASTER_PORT=12345
MASTER_ADDR="localhost"
NODE_RANK=0
NNODES=1
ENABLE_PROFILER="False"
TRAIN_STOP_STEP=1000        # train early stop 步数
SAMPLE_STATISTICS_NUM=100   # latency采集步数
NPROC_PER_NODE=$(echo $DEVICE_ID | awk -F',' '{print NF}')


torchrun \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    run.py \
    --config_file overall/LLM_deepspeed_npu.yaml HLLM/HLLM.yaml \
    --enable_profiler ${ENABLE_PROFILER}  \
    --train_stop_step ${TRAIN_STOP_STEP} \
    --sample_statistics_num ${SAMPLE_STATISTICS_NUM} \
    --use_item_cache True

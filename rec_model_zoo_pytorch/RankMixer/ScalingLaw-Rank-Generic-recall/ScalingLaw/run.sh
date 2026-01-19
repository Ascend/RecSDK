#!/bin/bash
echo -e "\n--> Generative Recommenders on ROMA environment\n"


echo -e "Period: ${period}"

date=${period%-*}
echo "Current Traing Period is ${date}"

sitepkgs_dir=$(pip show torch |grep -w 'Location:' | awk '{print $2}')
echo "site-packages dir: $sitepkgs_dir"
# 配置CANN相关环境变量（根据环境自行配置）
# source .../setenv.bash


unset ASCEND_CUSTOM_OPP_PATH
cur_dir=$(dirname $0)
train_workdir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname "$train_workdir")
echo ${train_workdir}


file="${sitepkgs_dir}/fbgemm_gpu/split_table_batched_embeddings_ops_training.py"
# 配置device， 先检查是否已经存在 "NPU = 3"
if ! grep -q "NPU = 3" "$file"; then
    sed -i '/CUDA = 1/a\    NPU = 3' "$file"
    echo "Inserted 'NPU = 3' into $file"
else
    echo "'NPU = 3' already exists in $file, skip insertion."
fi
  echo "====== Install mxrec ops finished ======"
# fi

param="${1:-npu}"
if [ "$param" = "gpu" ]; then
    echo "run on gpu!!!"
    export NPU_FLAG=False
else
    echo "run on npu!!!"
    export NPU_FLAG=True
fi

while [[ "$#" -gt 1 ]]; do
    case $2 in
        --config_file=*) config_file="${1#*=}"; shift ;;
        --data_dir=*) data_dir="${1#*=}"; shift ;;
        --save_dir=*) save_dir="${1#*=}"; shift ;;
        *) echo "Unknown parameter passed: $2"; exit 1 ;;
    esac
done

# ===== 打印原始解析结果 =====
echo "Before defaults:"
echo "  config_file=${config_file:-<unset>}"
echo "  data_dir=${data_dir:-<unset>}"
echo "  save_dir=${save_dir:-<unset>}"

# ===== 设置默认值 =====
: "${config_file:=../sample_configs/train_amazon_books_0.2B.ini}" # 可以修改为不同大小的模型，默认0.2B
: "${data_dir:=../../amzn_books.ori}"  # 须手动改为数据集路径
: "${save_dir:=$parent_dir/modelfile/}"   # 改为预训练模型对应路径
export LOAD_PRETRAIN_MODEL=0  # 是否加载预训练模型，0不加载，1加载(精度校验时使用)

# 一级流水
export TASK_QUEUE_ENABLE=1


MASTER_HOST="${VC_WORKER_HOSTS:-localhost}"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
NNODES="${MA_NUM_HOSTS:-1}"
NGPUS_PER_NODE="${MA_NUM_GPUS:-1}"
NODE_RANK="${VC_TASK_INDEX:-0}"

export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
# export TORCH_DEVICE_BACKEND_AUTOLOAD=0
MASTER_PORT="12356"

VISIBLE_DEVICES=0
export ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}

export NNODES=${NNODES}

# Define the Python script
PYTHON_SCRIPT=${train_workdir}/main.py
PYTHON_ARGS="
    --config_file=${config_file} \
    --data_dir=${data_dir} \
    --save_dir=${save_dir}/modelfile/ \
    --is_train=False \
    --save_user_emb=False \
    --get_infer_result=False \
    --tensorboard_log_dir=${save_dir}/ScalingLaw/runs/exp_amzn_all/"
#按需添加--eval_batch_size=128修改batch size
export PYTHONPATH=$(dirname $(realpath $(find ${cur_dir} -name "DLRM.py"))):${PYTHONPATH}

#megatron开关，True则启用， False则使用原本的torchrec框架
PYTHON_MEGATRON="--open_megatron False"

#megatron args。$MEGA_ARG为必备参数，没有会报错。需要与config里的手动同步
MICRO_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * NGPUS_PER_NODE * NNODES))
MEGA_ARGS="
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --num-layers 16 \
    --hidden-size 128 \
    --num-attention-heads 8 \
    --seq-length 100 \
    --max-position-embeddings 8000 \
    --use-tokenizer False \
    --optimizer adam \
"
MEGA_OPT_ARGS="
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
"

echo "$LOCAL_RANK"

# inductor相关环境变量
export ENABLE_COMPILE=0
export ENABLE_GRAPH=0
export ENABLE_SYNC_LOADER=1
export CHECK_PRECISION=0
export TORCHNPU_PRECOMPILE_THREADS=64 # 预编译线程数

# catlass相关环境变量
export CATLASS_EVG_FUSION=1
export CATLASS_EPOILOGUE_FUSION=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=CATLASS,ATen
export TORCHINDUCTOR_NPU_CATLASS_DIR=/xxx/catlass
export TORCHINDUCTOR_CATLASS_ENABLED_OPS="mm,addmm,bmm,grouped_mm"

export PROFILING_FLAG=1
if [ "$PYTHON_MEGATRON" = "--open_megatron True" ];then
    echo "========启用megatron=========="
    torchrun --nproc_per_node=$NGPUS_PER_NODE \
         --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         $PYTHON_SCRIPT \
         $PYTHON_MEGATRON \
         $PYTHON_ARGS \
         $MEGA_ARGS  
else
    echo "========启用torchrec=========="
    torchrun --nproc_per_node=$NGPUS_PER_NODE \
         --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         $PYTHON_SCRIPT \
         $PYTHON_ARGS 
fi
#!/bin/bash
echo -e "Period: ${period}"

date=${period%-*}
echo "Current Traing Period is ${date}"

sitepkgs_dir=$(pip show torch |grep -w 'Location:' | awk '{print $2}')
echo "site-packages dir: $sitepkgs_dir"
# Source setenv.bash here if the runtime has not initialized CANN.

unset ASCEND_CUSTOM_OPP_PATH
cur_dir=$(dirname $0)
train_workdir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname "$train_workdir")
echo ${train_workdir}
file="${sitepkgs_dir}/fbgemm_gpu/split_table_batched_embeddings_ops_training.py"

param="npu"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    param="$1"
    shift
fi

mode="${MODEL_MODE:-train}"
is_train=True
save_user_emb=False
case "${mode}" in
    train)
        ;;
    eval|infer|inference)
        is_train=False
        ;;
    save_user_emb)
        is_train=False
        save_user_emb=True
        ;;
    *)
        echo "ERROR: unsupported MODEL_MODE=${mode}" >&2
        exit 1
        ;;
esac

if [ "$param" = "gpu" ]; then
    echo "run on gpu!!!"
    export NPU_FLAG=False
else
    echo "run on npu!!!"
    export NPU_FLAG=True
fi
echo "Runtime mode: ${mode}, is_train=${is_train}, save_user_emb=${save_user_emb}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_file=*) config_file="${1#*=}"; shift ;;
        --data_dir=*) data_dir="${1#*=}"; shift ;;
        --save_dir=*) save_dir="${1#*=}"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# ===== Print user supplied arguments =====
echo "Before defaults:"
echo "  config_file=${config_file:-<unset>}"
echo "  data_dir=${data_dir:-<unset>}"
echo "  save_dir=${save_dir:-<unset>}"

data_dir_was_default=0
if [ -z "${data_dir:-}" ]; then
    data_dir_was_default=1
fi

# ===== Default runtime paths =====
: "${config_file:=../sample_configs/train_amazon_books_1B.ini}"
: "${data_dir:=../../amzn_books.ori}"
: "${save_dir:=$parent_dir/modelfile/}"
: "${LOAD_PRETRAIN_MODEL:=0}"
export LOAD_PRETRAIN_MODEL

if [ "${data_dir_was_default}" = "1" ]; then
    if [ -L "${data_dir}" ]; then
        rm -f "${data_dir}"
    elif [ -e "${data_dir}" ] && [ ! -d "${data_dir}" ]; then
        rm -f "${data_dir}"
    fi
fi

# If the default data directory is missing, try common benchmark dataset locations.
if [ ! -d "${data_dir}" ]; then
    if [ "${data_dir_was_default}" = "1" ]; then
        auto_data_sources=(
            "${train_workdir}/../../../../../../../datasets"
            "${train_workdir}/../../../../../../../dataset"
            "${train_workdir}/../../../../../../../../datasets"
            "${train_workdir}/../../../../../../../../dataset"
        )
        for src in "${auto_data_sources[@]}"; do
            if [ -d "${src}" ]; then
                mkdir -p "$(dirname "${data_dir}")"
                mkdir -p "${data_dir}"
                echo "Auto copying data_dir ${src} -> ${data_dir}"
                cp -r "${src}/." "${data_dir}"
                break
            fi
        done
    fi
fi

if [ ! -d "${data_dir}" ]; then
    echo "ERROR: data_dir does not exist: ${data_dir}" >&2
    echo "Pass --data_dir=/path/to/amzn_books.ori or place data under rec_model_zoo_pytorch/tokenmixer-large/amzn_books.ori." >&2
    exit 1
fi

if ! compgen -G "${data_dir}/*.csv" > /dev/null; then
    echo "ERROR: data_dir has no *.csv files: ${data_dir}" >&2
    echo "Provide real Amazon Books CSV data under tokenmixer-large/amzn_books.ori or pass --data_dir=/path/to/amzn_books.ori." >&2
    exit 1
fi

: "${TASK_QUEUE_ENABLE:=2}"
export TASK_QUEUE_ENABLE
: "${TP_ALL_GATHER_INTO_TENSOR:=1}"
export TP_ALL_GATHER_INTO_TENSOR
: "${TP_COMM_OVERLAP:=1}"
export TP_COMM_OVERLAP

MASTER_HOST="${VC_WORKER_HOSTS:-localhost}"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
NNODES="${MA_NUM_HOSTS:-1}"
NODE_RANK="${VC_TASK_INDEX:-0}"
MASTER_PORT="${MASTER_PORT:-12356}"

# Python entrypoint
PYTHON_SCRIPT=${train_workdir}/main.py
PYTHON_ARGS="
    --config_file=${config_file} \
    --data_dir=${data_dir} \
    --save_dir=${save_dir}/modelfile/ \
    --is_train=${is_train} \
    --save_user_emb=${save_user_emb} \
    --get_infer_result=False \
    --tensorboard_log_dir=${save_dir}/ScalingLaw/runs/exp_amzn_all/"

export PYTHONPATH=$(dirname $(realpath $(find ${cur_dir} -name "DLRM.py"))):${PYTHONPATH}
echo "$LOCAL_RANK"
visible_devices_was_set=0
if [ -n "${VISIBLE_DEVICES+x}" ]; then
    visible_devices_was_set=1
fi

# ===== Runtime toggles =====
# export MODEL_PROFILING_FLAG=true  # enable torch/NPU profiling when needed
# export MODEL_EPOCH=1              # override epoch count for quick checks
: "${SAVE_MODEL_FLAG:=0}"
: "${SAMPLE_STATISTICS_NUM:=10}"
export SAVE_MODEL_FLAG
export SAMPLE_STATISTICS_NUM
: "${VISIBLE_DEVICES:=0}"

config_use_token_parallel="$(grep -m1 '"use_token_parallel"' "${config_file}" 2>/dev/null | sed -E 's/.*"use_token_parallel"[[:space:]]*:[[:space:]]*([^, #]+).*/\1/I')"
config_token_parallel_size="$(grep -m1 '"token_parallel_size"' "${config_file}" 2>/dev/null | sed -E 's/.*"token_parallel_size"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/')"
config_token_parallel_size="${config_token_parallel_size:-0}"

if [ "${visible_devices_was_set}" = "0" ]; then
    auto_device_count=0
    if [ -n "${MA_NUM_GPUS:-}" ] && [ "${MA_NUM_GPUS}" -gt 1 ]; then
        auto_device_count="${MA_NUM_GPUS}"
    elif [[ "${config_use_token_parallel}" =~ ^([Tt]rue|1)$ ]] && [ "${config_token_parallel_size}" -gt 1 ]; then
        auto_device_count="${config_token_parallel_size}"
    fi
    if [ "${auto_device_count}" -gt 1 ]; then
        visible_devices_auto=()
        for ((dev_id=0; dev_id<auto_device_count; dev_id++)); do
            visible_devices_auto+=("${dev_id}")
        done
        VISIBLE_DEVICES="$(IFS=,; echo "${visible_devices_auto[*]}")"
        echo "Auto set VISIBLE_DEVICES=${VISIBLE_DEVICES} for ${auto_device_count} local process(es)"
    fi
fi

if [ -n "${MA_NUM_GPUS:-}" ]; then
    NGPUS_PER_NODE="${MA_NUM_GPUS}"
else
    IFS=',' read -ra visible_device_array <<< "${VISIBLE_DEVICES}"
    NGPUS_PER_NODE="${#visible_device_array[@]}"
fi

if [[ "${config_use_token_parallel}" =~ ^([Tt]rue|1)$ ]] && [ "${config_token_parallel_size}" -gt 1 ]; then
    if [ "${config_token_parallel_size}" -gt "${NGPUS_PER_NODE}" ]; then
        echo "WARNING: Token Parallel is configured as size ${config_token_parallel_size}, but torchrun will start ${NGPUS_PER_NODE} process(es)." >&2
        echo "         Set VISIBLE_DEVICES=0,1,...,$((config_token_parallel_size - 1)) or MA_NUM_GPUS=${config_token_parallel_size} to enable TP." >&2
    elif [ $((NGPUS_PER_NODE % config_token_parallel_size)) -ne 0 ]; then
        echo "WARNING: Token Parallel size ${config_token_parallel_size} should divide local process count ${NGPUS_PER_NODE}." >&2
    fi
fi
export ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}

if [ "${NPU_FLAG}" = "True" ]; then
    : "${PYTORCH_NPU_ALLOC_CONF:=expandable_segments:False}"
    : "${HCCL_OVERLAPPING:=1}"
    : "${HCCL_FUSION:=1}"
    : "${HCCL_BUFFSIZE:=128}"
    : "${NPU_DATALOADER_START_METHOD:=spawn}"
    : "${NPU_DATALOADER_PIN_MEMORY:=0}"
    export PYTORCH_NPU_ALLOC_CONF
    export HCCL_OVERLAPPING
    export HCCL_FUSION
    export HCCL_BUFFSIZE
    export NPU_DATALOADER_START_METHOD
    export NPU_DATALOADER_PIN_MEMORY
fi

echo "======== start torchrec =========="
torchrun --nproc_per_node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $PYTHON_SCRIPT \
        $PYTHON_ARGS

#!/bin/bash
DEVICE_ID=0
DEVICE=npu # cpu/npu/cuda
PREPROCESSED_DATASET= #xxx/criteo 数据集路径
MODELS=("dlrm")
MODE=train # train/eval/infer
OPEN_PROFILING=true 
export JOB_ID=10085

if [ $DEVICE == "npu" ]; then
    export ASCEND_DEVICE_ID=${DEVICE_ID}
    export RANK_ID=${DEVICE_ID}
    export DEVICE_ID=${DEVICE_ID}
    echo "set npu"
fi

echo "use "${DEVICE}:${DEVICE_ID}

for model in ${MODELS[@]}; do
    python3 ${model}.py --device ${DEVICE} \
                        --device_id=${DEVICE_ID} \
                        --mode=${MODE}\
                        --data_dir=$PREPROCESSED_DATASET \
                        --model_dir=../checkpoint/criteo/ \
                        --profiling_mode=${OPEN_PROFILING} \
                        --profiling_path=./profiling/${model}/${MODE} \
                        --compile=false \
                        --num_epochs=1 \
                        --batch_size=4096 \
                        --learning_rate=0.001 \
                        --embedding_size=16
done
echo "finished!"
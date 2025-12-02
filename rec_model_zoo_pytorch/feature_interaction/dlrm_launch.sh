#!/bin/bash
DEVICE_ID=0
DEVICE=cuda # cpu/npu/cuda
PREPROCESSED_DATASET=/home #xxx/criteo 数据集路径
MODELS=("dlrm")
MODE=test_qps # train/eval/test/test_qps, test_qps为生成的虚拟数据
OPEN_PROFILING=true
TEST_BATCH_SIZE=(128)
export JOB_ID=10085

if [ $DEVICE == "npu" ]; then
    export ASCEND_DEVICE_ID=${DEVICE_ID}
    export RANK_ID=${DEVICE_ID}
    export DEVICE_ID=${DEVICE_ID}
    echo "set npu"
fi

echo "use "${DEVICE}:${DEVICE_ID}

for model in ${MODELS[@]}; do
    for batch_size in ${TEST_BATCH_SIZE[@]}; do
        python ${model}.py --device ${DEVICE} \
                            --device_id=${DEVICE_ID} \
                            --mode=${MODE} \
                            --data_dir=$PREPROCESSED_DATASET \
                            --model_dir=../checkpoint/criteo/ \
                            --report_dir=../reports \
                            --profiling_mode=${OPEN_PROFILING} \
                            --profiling_path=../profiling \
                            --num_epochs=1 \
                            --batch_size=128\
                            --learning_rate=0.001 \
                            --embedding_size=8 \
                            --hf32=true \
                            --graph=true \
                            --compile=false \
                            --test_batch_size=${batch_size} # MODE=test_qps时改这个设置batch_size
    done
done
echo "finished!"
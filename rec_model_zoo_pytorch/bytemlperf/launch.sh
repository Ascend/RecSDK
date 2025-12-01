#!/bin/bash
export PRE_GRAPH_OPTIMIZER=1

DEVICE_ID=0
DEVICE=cuda # cpu/npu/cuda
PREPROCESSED_DATASET= #xxx/aliccp_out 数据集路径
MODELS=("resnet50")
MODE=test_qps # train/eval/test/test_qps/get_layer_result, test_qps为生成的虚拟数据
OPEN_PROFILING=true
TEST_BATCH_SIZE=(16)
MODEL_PATH=
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
        python3 ${model}.py --device ${DEVICE} \
                            --device_id=${DEVICE_ID} \
                            --mode=${MODE} \
                            --data_dir=$PREPROCESSED_DATASET \
                            --model_dir=${MODEL_PATH} \
                            --report_dir=../reports/ \
                            --profiling_mode=${OPEN_PROFILING} \
                            --profiling_path=../profiling/ \
                            --num_epochs=1 \
                            --learning_rate=0.001 \
                            --embedding_size=32 \
                            --hf32=true \
                            --graph=false \
                            --compile=false \
                            --dynamic_batch=false \
                            --test_batch_size=${batch_size} # MODE=test_qps时改这个设置batch_size
    done
done

echo "finished!"
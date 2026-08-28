DEVICE_ID=0
DEVICE=cuda # cpu/npu/cuda

PREPROCESSED_DATASET= #xxx/training_data_all-rlData.csv 数据集路径
test_DATASET= #xxx/period-7.csv 测试数据集路径
save_path=./save_model  #训练权重保存路径

MODE=train  #train/test
OPEN_PROFILING=true
export JOB_ID=10085
export SHAPE_LIST="" # "batch_size,seq_len;batch_size,seq_len;..."

if [ $DEVICE == "npu" ]; then
    export ASCEND_DEVICE_ID=${DEVICE_ID}
    export RANK_ID=${DEVICE_ID}
    export DEVICE_ID=${DEVICE_ID}
    echo "set npu"
fi

echo "use "${DEVICE}:${DEVICE_ID}

python3 run_decision_transformer.py  --device ${DEVICE} \
                                    --device_id=${DEVICE_ID} \
                                    --mode=${MODE} \
                                    --data_dir=$PREPROCESSED_DATASET \
                                    --test_dir=$test_DATASET \
                                    --modelsave_dir=$save_path \
                                    --report_dir=./reports/ \
                                    --profiling_mode=${OPEN_PROFILING} \
                                    --profiling_path=./profiling \
                                    --batch_size=128 \
                                    --learning_rate=0.001 \
                                    --step_num=210 \
                                    --hf32=true \
                                    --graph=false \
                                    --compile=false \
                                    --enable_dynamic_compile False \
                                    --check_results=false \
                                    --dynamic_batch=false \

echo "finished!"

#!/bin/bash
set -e
#---------------------------------------------
# lib relate
#---------------------------------------------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#---------------------------------------------
# train job related
#---------------------------------------------
export OMP_NUM_THREADS=12
export WORLD_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 数据集位置，根据实际情况修改
export PREPROCESSED_DATASET="/path/to/data"
# 算子适配文件.so路径，根据实际情况修改
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"

export TOTAL_TRAINING_SAMPLES=4195197692
export GLOBAL_BATCH_SIZE=16384
export LIMIT_TRAIN_BATCHES=2000
export LIMIT_TEST_BATCHES=1000
FEATURE_NUM=$((40000000 * ${WORLD_SIZE} / 8))

function run_dlrm_model(){
  torchx run -s local_cwd dist.ddp -j 1x${WORLD_SIZE} --script dlrm_main.py -- \
    --epochs 1 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
    --in_memory_binary_criteo_path $PREPROCESSED_DATASET \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --limit_train_batches $LIMIT_TRAIN_BATCHES \
    --limit_test_batches $LIMIT_TEST_BATCHES \
    --num_embeddings_per_feature ${FEATURE_NUM},39060,17295,7424,20265,3,7122,1543,63,${FEATURE_NUM},3067956,405282,10,2209,11938,155,4,976,14,${FEATURE_NUM},${FEATURE_NUM},${FEATURE_NUM},590152,12973,108,36 \
    --embedding_dim 128 \
    --multi_hot_distribution_type uniform \
    --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --adagrad \
    --learning_rate 0.005 \
    --pin_memory \
    --mmap_mode \
    2>&1 |tee ${model}_use_ec_${use_ec}_$(date '+%Y%m%d_%H%M%S').log
}

MODES=("torchrec" "hybrid_torchrec")

for model in "${MODES[@]}"; do
  # 重置环境变量
  export WITH_TORCHREC=0
  export WITH_HYBRID_TORCHREC=0
  # 设置当前模式
  case $model in
    "torchrec") export WITH_TORCHREC=1 ;;
    "hybrid_torchrec") export WITH_HYBRID_TORCHREC=1 ;;
  esac
  echo "MODEL_TYPE: $model "
  run_dlrm_model # 执行模型
  sleep 5
done



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
export WORLD_SIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0

#---------------------------------------------
# train with ec
#---------------------------------------------
# 仅使用embcache模式运行模型时才支持EC，可设置USE_EC=1,DO_EC_LOCAL_UNIQUE=1其他场景环境变量不起作用。
# export USE_EC=1
# export DO_EC_LOCAL_UNIQUE=0

# 数据集位置，根据实际情况修改
export PREPROCESSED_DATASET=""

export TOTAL_TRAINING_SAMPLES=4195197692
export GLOBAL_BATCH_SIZE=16384
export LIMIT_TRAIN_BATCHES=200000
export LIMIT_TEST_BATCHES=1000
export WITH_EMBCACHE_AND_SAVE=0
export WITH_EMBCACHE_AND_LOAD=1
export EXECUTE_TRAIN=0
export EXECUTE_SING_CARD_EVAL=1
export EMBCACHE_SIZE_ON_DEVICE_MEM=8589934592
export GLOG_stderrthreshold=1
export ENABLE_FAST_HASHMAP=1
export USE_EC=1
export DO_EC_LOCAL_UNIQUE=0
export OMP_NUM_THREADS=2
FEATURE_NUM=$((40000000 / 2))

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
    2>&1 |tee ${model}_use_ec_${USE_EC}_$(date '+%Y%m%d_%H%M%S').log
}

MODES=("embcache")

for model in "${MODES[@]}"; do
  # 重置环境变量
  export WITH_TORCHREC=0
  export WITH_HYBRID_TORCHREC=0
  export WITH_EMBCACHE=0
  # 设置当前模式
  case $model in
    "torchrec") export WITH_TORCHREC=1 ;;
    "hybrid_torchrec") export WITH_HYBRID_TORCHREC=1 ;;
    "embcache") export WITH_EMBCACHE=1 ;;
  esac
  echo "MODEL_TYPE: $model "
  run_dlrm_model # 执行模型
  sleep 5
done

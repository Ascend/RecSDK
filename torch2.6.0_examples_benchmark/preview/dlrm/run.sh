#!/bin/bash
set -e
#---------------------------------------------
# lib relate
#---------------------------------------------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_PRELOAD=/usr/lib64/libgomp.so.1
export LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/embedding_cache/:$LD_LIBRARY_PATH

#---------------------------------------------
# ascend related
#---------------------------------------------
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# local unique
export DO_LOCAL_UNIQUE=0

#---------------------------------------------
# embcache related
#---------------------------------------------
export USE_MOD_BUCKETIZE=1 #EMBCACHE模式下采用MOD方式分桶
export EMBCACHE_SIZE_ON_HBM=$((16*1024*1024*1024))

export SPARSE_OPTIM_NUM=1
export ENABLE_GLOBAL_UNIQUE=0
# ENABLE_FAST_HASHMAP=false时,默认适用unordered_map
export ENABLE_FAST_HASHMAP=1
export FAST_HASHMAP_RESERVE_BUCKET_NUM=$((2*1024*1024))

# FAST_HASHMAP will use EMB_MEMORY_POOL
export EMB_MEMORY_POOL_THREAD_NUM=4
export EMB_MEMORY_POOL_SIZE=102400

# 0:INFO; 1:WARNINH; 2:ERROR; 3:FATAL; (on profiling, we set 0; otherwise we set 1)
export GLOG_MIN_LOG_LEVEL=1


export USE_EC=1
#---------------------------------------------
# train job related
#---------------------------------------------
# data path
export PREPROCESSED_DATASET=/mxrec_disk1/Criteo_all/Criteo_temp/temp_output_of_step_1  #根据实际情况修改
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"                     #根据实际情况修改
export OMP_NUM_THREADS=12
export TOTAL_TRAINING_SAMPLES=4195197692
export GLOBAL_BATCH_SIZE=16384
export LIMIT_TRAIN_BATCHES=2000
export LIMIT_TEST_BATCHES=1000

export WORLD_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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


MODES=("torchrec" "hybrid_torchrec" "embcache")
USE_EC_VALUES=(0 1)

for model in "${MODES[@]}"; do
  for use_ec in "${USE_EC_VALUES[@]}"; do
    # 重置环境变量
    export WITH_TORCHREC=0
    export WITH_HYBRID_TORCHREC=0
    export WITH_EMBCACHE=0
    export USE_EC=$use_ec

    # 设置当前模式
    case $model in
      "torchrec") export WITH_TORCHREC=1 ;;
      "hybrid_torchrec") export WITH_HYBRID_TORCHREC=1 ;;
      "embcache") export WITH_EMBCACHE=1 ;;
    esac
    echo "MODEL_TYPE: $model (use_ec=$use_ec)"
    run_dlrm_model # 执行模型
    sleep 5
  done
done


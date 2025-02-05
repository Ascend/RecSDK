#!/bin/bash

source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/tfplugin/set_env.sh
export JOB_ID=10089
export ASCEND_DEVICE_ID=0
export RANK_ID=0
export DEVICE_ID=0

export PREPROCESSED_DATASET=/home/ma-user/work/ydz/EffBench/feature_interaction/data/criteo/

models=("lr" "fm" "widedeep" "ipnn" "opnn" "pnn" "deepfm" "autoint" "autoint_plus" "fibinet" "dcnv2")


for model in ${models[@]}; do
python3 -u ./models/${model}.py --data_dir=$PREPROCESSED_DATASET --model_dir=../checkpoint/criteo/${model}/ \
        --task_type=profiling_infer --field_size=39 --feature_size=2100000 --train_size=33003326 --embedding_size=10
done

python3 -u ./models/ffm.py --data_dir=$PREPROCESSED_DATASET --model_dir=../checkpoint/criteo/ffm/ \
         --task_type=profiling_infer --field_size=39 --feature_size=2100000 --train_size=33003326 --embedding_size=2

python3 -u ./models/afn.py --data_dir=$PREPROCESSED_DATASET --model_dir=../checkpoint/criteo/afn/ \
         --task_type=profiling_infer --field_size=39 --feature_size=2100000 --train_size=33003326 --embedding_size=10 --hidden_size=1500

python3 -u ./models/afn_plus.py --data_dir=$PREPROCESSED_DATASET --model_dir=../checkpoint/criteo/afn_plus/ \
         --task_type=profiling_infer --field_size=39 --feature_size=2100000 --train_size=33003326 --embedding_size=10 --hidden_size=1500


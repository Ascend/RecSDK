#!/bin/bash

source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/tfplugin/set_env.sh
export JOB_ID=10085
export ASCEND_DEVICE_ID=0
export RANK_ID=0
export DEVICE_ID=0

export PREPROCESSED_DATASET=/home/ma-user/work/EffBench/behaviour_and_multi_task/data/aliccp/aliccp_out/

models=("din" "bst" "eta" "can" "dffm" "esmm" "sharedbottom" "mmoe" "ple" "dmt")



for model in ${models[@]}; do
    python3 -u ./models/${model}.py --task_type=profiling_train \
                                    --data_dir=$PREPROCESSED_DATASET \
                                    --max_seq_len=50
done

#!/bin/bash

cur_path=$(dirname "$(readlink -f "$0")")
project_root=$(cd "$cur_path/../../../.." && pwd)

source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/tfplugin/set_env.sh
export JOB_ID=10084
export ASCEND_DEVICE_ID=0
export RANK_ID=0
export DEVICE_ID=0

export PREPROCESSED_DATASET=/home/ma-user/work/ydz/EffBench/behaviour_and_multi_task/data/aliccp/aliccp_out/
export PYTHONPATH=${project_root}:$PYTHONPATH

models=("din" "bst" "eta" "can" "dffm" "esmm" "sharedbottom" "mmoe" "ple" "dmt")

for model in ${models[@]}; do
    python3 -u ./models/${model}.py --task_type=train \
                                    --data_dir=$PREPROCESSED_DATASET \
                                    --max_seq_len=50
done

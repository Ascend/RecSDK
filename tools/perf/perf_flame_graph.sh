#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
# Description: performace analysis tool
# Author: MindSDK
# Create: 2023
# History: NA

#set -x

curr_path=$(cd $(dirname $0); pwd)

LOG_INFO() { echo -e "\033[1;4;32m$1\033[0m" ; }
LOG_NOTICE() { echo -e "\033[1;4;45m$1\033[0m" ; }
LOG_WARN() { echo -e "\033[1;31m[WARN]$1\033[0m" ; }
LOG_ERROR() { echo -e "\033[1;31m[Error]$1\033[0m" ; }

# ---------------config start---------------------
model_run_path=/path/to/model/run
run_cmd="bash run.sh"
flame_graph_path=/home/FlameGraph
# ---------------config end---------------------

cd "${model_run_path}"
rm -rf perf*

#---- perf cpu-clock on all workers and build flame graph------------
perf record -F 99 -a -g  "${run_cmd}"
wait $!

perf script -i perf.data | \
  "${flame_graph_path}"/stackcollapse-perf.pl | \
  "${flame_graph_path}"/flamegraph.pl > perf_mxRec.svg
wait $!

LOG_INFO "perf_mxRec.svg is created, please check!"



#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
# Description: performace analysis tool
# Author: MindX SDK
# Create: 2023
# History: NA

#set -x

LOG_INFO() { echo -e "\033[1;4;32m$1\033[0m" ; }
LOG_NOTICE() { echo -e "\033[1;4;45m$1\033[0m" ; }
LOG_WARN() { echo -e "\033[1;31m[WARN]$1\033[0m" ; }
LOG_ERROR() { echo -e "\033[1;31m[Error]$1\033[0m" ; }

logfile=$1

# ---------------config start---------------------
batchsize=9600
parallel=8
nv_throughput=820000
# ---------------config end---------------------

validate_options()
{
  if [ $# -ne 1 ]; then
    LOG_ERROR "NO log_file"
    echo "[Usage]: bash $0 your_file.log"
    exit 1
  fi
}

print_throughput()
{
  LOG_INFO "=========Throughput====================="
  nv_sps=$(awk 'BEGIN{printf "%.2f\n",('${nv_throughput}'/'$batchsize'/'$parallel')}')
  LOG_NOTICE "batchsize:${batchsize}, parallel:${parallel}"
  LOG_NOTICE "nv_throughput:${nv_throughput}, nv_sps:${nv_sps}"

  grep 'tensorflow:global_step/sec' $logfile | \
    awk -F" " '{sum+=$NF} END \
    {printf "Throughput: avg=%0.3f, xA100:%0.3f\n", \
    sum/NR, sum/NR/'${nv_sps}'}'

  grep 'tensorflow:global_step/sec' $logfile | \
    awk -F" " 'BEGIN {sum=0; count=0;} {if ($NF > 3) {sum+=$NF; count++;}} END \
    {printf "Throughput: after filter(<3), avg=%0.3f, xA100:%0.3f\n", \
    sum/count, sum/count/'${nv_sps}'}'

  grep 'tensorflow:global_step/sec' $logfile | \
    awk -F" " 'BEGIN {max=0} {if($2>max) max=$2} END \
    {printf "Throughput: max=%0.3f, xA100:%0.3f\n", max, max/'${nv_sps}'}'
}

main()
{
  validate_options $@
  print_throughput
}

main $@

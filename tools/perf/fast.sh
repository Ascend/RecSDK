#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
# Description: performace analysis tool
# Author: MindX SDK
# Create: 2023
# History: NA

# -----------------------------------------ReadMe  Begin--------------------------------------------
# 1. 功能描述
# 本工具用来分析模型执行过程中pipeline中各个pipe的耗时、以及各个pipe中的子模块(Step)的耗时，以便于发现系统瓶颈。
# (pipeline的基本原理是：每个pipe的耗时近似相等，pipe之间的耗时能够互相掩盖起来，这样，才能减少堵塞和等待，提升吞吐。)
#
# 2. 使用方法
# bash fast.sh your_log_file.log
#
# 3. 注意事项
# 基于spdlog::info，mxRec中添加了TimeCost打点日志，因此，在执行前务必确保run.sh中设置
# SPDLOG_LEVEL=info 或者 SPDLOG_LEEL=debug  (如果没有设置，本工具会退出，并给予提示)
#
# 4. 解读结果
# (1) Pipeline: 整个Pipeline由多个Pipe串行构成，性能分析结果分Pipe呈现，例如Pipe-1/Pipe-2/Pipe-3/Pipe-4等；
# (2) Pipe: 每个Pipe级都会有一个整个耗时。(我们希望每个Pipe的耗时近似相等，这样Pipe之间才能互相掩盖，流水线效率才最高)
# (3) 子模块(Step)：一个Pipe可能有多个串行的子模块(Step)构成、子模块又可能包含下一级子模块(SubStep)。因此，在性能分级报告中，
# 下一级的子模块耗时用--开头，再下一级的子模块耗时用----开头；依次类推；(上一级的耗时中包含了下一级的耗时)
#
# 5. 性能调优
# 通过分析报告，我们可能会发现：
# (1)耗时特别长的Pipe；
# (2)耗时特别长的子模块；
# 需要具体问题具体分析，针对性的调优或者开展深度优化。
# 例如：如果发现Tensorflow数据解析慢(Pipe-1)，导致供应不足，可以调节Tensorflow侧解析数据的num_parallel参数；
# 如果发现CPU打满而导致数据预处理阻塞(Pipe 2: Data Preprocess)，则可以调低KEY_PROCESS_THREAD_NUM (默认为6);
# 如果发现H2D阻塞(Pipe 4: H2D Send Tensors (no DDR)),则可能需要排查NPU侧GetNext或者DNN训练是否堵塞。
# 然而，对于一些深层的问题，可能涉及到需要开展深度优化：比如Pipe拆分、串行改并行、锁优化、执行逻辑调整。
# 另外，本工具也可以作为性能优化的参考，例如优化了某个子模块，可以对比观察(优化前vs优化后)该子模块的耗时，
# 同时对比观察端到端耗时、吞吐变化等。
#
# 6. 该工具也需要不断升级，和代码同步更新，欢迎大家修改、完善。Good Luck!
# -----------------------------------------ReadMe  End--------------------------------------------
#set -x

LOG_INFO() { echo -e "\033[1;4;32m$1\033[0m" ; }
LOG_NOTICE() { echo -e "\033[1;4;45m$1\033[0m" ; }
LOG_WARN() { echo -e "\033[1;31m[WARN]$1\033[0m" ; }
LOG_ERROR() { echo -e "\033[1;31m[Error]$1\033[0m" ; }

logfile=$1

validate_options()
{
  if [ $# -ne 1 ]; then
    LOG_ERROR "NO log_file"
    echo "[Usage]: bash $0 log_file"
    exit 1
  fi
}

check_spdlog_level()
{
  $(grep 'ReadEmbKeyV2Static' $logfile > /dev/null 2>&1)
  if [ $? != 0 ]; then
    $(grep 'ReadEmbKeyV2Dynamic' $logfile > /dev/null 2>&1)
    if [ $? != 0 ]; then
        LOG_ERROR "No timecost-related logs, please check 'mpi_args' in your run.sh,
                make sure SPDLOG_LEVEL=info or SPDLOG_LEEL=debug, and run again!"
        exit 1
    fi
  fi
}

parse_pipe_1_data_parser()
{
  LOG_NOTICE "Pipe-1: Data Parser"

  $(grep 'ReadEmbKeyV2Dynamic' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    LOG_INFO "Step-1.x ReadEmbKeyV2 Dynamic"
  else
    LOG_INFO "Step-1.x ReadEmbKeyV2 Static"
  fi

  grep 'read batch cost(ms)' $logfile | cut -d" " -f7| \
    awk -F "[:,]" '{sum+=$2} END {printf "read batch cost: avg=%0.1f\n", sum/NR}'

  grep 'enqueueTC(ms)' $logfile | grep -v 'timeout' | cut -d" " -f11 | \
    awk -F "[:,]" '{sum+=$2} END {printf "--|enqueueTC: avg=%0.1f\n", sum/NR}'

  grep 'elapsed from last(ms)' $logfile | grep -v 'timeout' | cut -d" " -f10 | \
    awk -F "[:,]" '{print $2}' | \
    awk 'BEGIN {sum=0; count=0} {if($1<1000) {sum+=$NF; count++} } END \
    {printf "elapsed from last: avg=%0.1f\n", sum/count}'
}

parse_pipe_2_key_process()
{
  LOG_NOTICE "Pipe-2: Data Preprocess"

  grep 'getAndProcessTC(ms)' $logfile | cut -d" " -f5 | \
    awk -F"[:,]" '{print $2}' | \
    awk 'BEGIN{count=0; total=0;} {if ($1<2000) {total+=$NF; count++;}} END \
      {printf "getAndProcessTC(filter>2000ms): avg=%0.3f\n", total/count}'

  LOG_INFO "Step-2.1 GetBatchData"

  grep 'getBatchDataTC' $logfile | \
    awk -F":" 'BEGIN { max=0 } { sum+=$NF; if($NF>max) max=$NF } END \
    {printf "--|get data time: total=%d, max=%0.1f, avg=%0.1f\n", NR, max, sum/NR}'

  grep 'getBatchDataTC' $logfile | \
    awk -F":" 'BEGIN {sum=0; count=0;} {if($NF<2000) {sum+=$NF; count++;}} END \
    {printf "--|get data time(filter>2000ms): count=%d, avg=%0.1f\n", count, sum/count}'

  grep 'getBatchDataTC' $logfile | \
    awk -F":" 'BEGIN { total=0; none_zero_ms_num=0 } { total++; if($NF>0) none_zero_ms_num++ } END \
      {printf "--|get data time: total=%d, none_zero_ms_num=%d, none_zero_ms_rate=%0.3f, zero_ms_rate=%0.3f\n", \
      total, none_zero_ms_num, none_zero_ms_num/total, (1-none_zero_ms_num/total)}'

  LOG_INFO "Step-2.2 KeyProcess"

  grep 'key process cost' $logfile | cut -d" " -f8 | cut -d ":" -f2 | cut -d"," -f1 | grep  '^[0-9]' | grep '[0-9]$' | \
    awk 'BEGIN {sum=0; count=0;} {if($NF<2000) {sum+=$NF; count++;}} END \
    {printf "--|key process cost(filter>2000): avg=%0.1f\n", sum/count}'

  # fast-unique related start
  $(grep 'ProcessBatchWithFastUnique(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'ProcessBatchWithFastUnique(ms)' $logfile | \
      awk -F":" '{sum+=$NF} END {printf "----|ProcessBatchWithFastUnique: avg=%0.1f\n", sum/NR}'
  fi

  $(grep 'FastUniqueCompute(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'FastUniqueCompute(ms)' $logfile | cut -d' ' -f4 | \
      awk -F"[:,]" '{sum+=$2} END {printf "------|FastUniqueCompute: avg=%0.1f\n", sum/NR}'
  fi

  $(grep 'GetScAll TimeCost(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'GetScAll TimeCost(ms)' $logfile | \
      awk -F":" '{sum+=$NF} END {printf "------|GetScAll: avg=%0.1f\n", sum/NR}'
  fi

  $(grep 'all2allTC TimeCost(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'all2allTC TimeCost(ms)' $logfile | \
      awk -F":" '{sum+=$NF} END {printf "------|all2allTC: avg=%0.1f\n", sum/NR}'
  fi
  # fast-unique related end

  $(grep 'UniqueTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'UniqueTC(ms)' $logfile | \
      awk -F":" '{sum+=$NF} END {printf "----|UniqueTC: avg=%0.1f\n", sum/NR}'
  fi

  $(grep 'processSplitKeysTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'processSplitKeysTC(ms)' $logfile | \
        awk -F":" '{sum+=$NF} END {printf "----|processSplitKeysTC: avg=%0.1f\n", sum/NR}'
  fi

  $(grep 'getScAllTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'getScAllTC(ms)' $logfile | \
      awk -F":" '{sum+=$NF} END {printf "------|getScAllTC(AllReduce-AllGather): avg=%0.1f\n", sum/NR}'
  fi

  $(grep 'uniqueAll2AllTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'uniqueAll2AllTC(ms)' $logfile | \
      awk -F":" '{sum+=$NF} END {printf "------|uniqueAll2AllTC(All2allv): avg=%0.1f\n", sum/NR}'
  fi

  $(grep 'buildRestoreVecTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'buildRestoreVecTC(ms)' $logfile | \
            awk -F":" '{sum+=$NF} END {printf "----|buildRestoreVecTC: avg=%0.1f\n", sum/NR}'
  fi

  # common start
  $(grep 'key2OffsetTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'key2OffsetTC(ms)' $logfile | \
      awk -F":" '{sum+=$NF} END {printf "----|key2OffsetTC: avg=%0.1f\n", sum/NR}'
  fi

  $(grep 'pushResultTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'pushResultTC(ms)' $logfile | \
      awk -F":"  '{sum+=$NF} END {printf "----|pushResultTC, avg=%0.1f\n", sum/NR}'
  fi
  # common end
}

parse_pipe_3_get_tensors_async_no_ddr()
{
  LOG_NOTICE "Pipe-3: Get Tensors async (no DDR)"

  $(grep 'getAllTensorTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
      grep 'getAllTensorTC(ms)' $logfile | \
        awk -F":"  '{sum+=$NF} END {print "getAllTensorTC, avg=", sum/NR}'
  fi
}

parse_pipe_4_send_tensors_async_no_ddr()
{
  LOG_NOTICE "Pipe-4: H2D Send Tensors async (no DDR)"

  $(grep 'sendTensorsTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'sendTensorsTC(ms)' $logfile | \
      awk -F":"  '{sum+=$NF} END {print "sendTensorsTC, avg=", sum/NR}'
  fi

  $(grep 'sendLookupTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
      grep 'sendLookupTC(ms)' $logfile | \
          awk -F":"  '{sum+=$NF} END {print "--|sendLookupTC, avg=", sum/NR}'
  fi

  $(grep 'sendRestoreTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
    grep 'sendRestoreTC(ms)' $logfile | \
      awk -F":"  '{sum+=$NF} END {print "--|sendRestoreTC, avg=", sum/NR}'
  fi
}

parse_pipe_3_get_and_send_tensors_with_ddr()
{
  LOG_NOTICE "Pipe-3: Get and Send Tensors (with DDR)"

  $(grep 'getAndSendTensorsTC(ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
      grep 'getAndSendTensorsTC(ms)' $logfile | \
            awk -F":"  '{sum+=$NF} END {print "GetAndSendTensors, avg=", sum/NR}'
  fi
}

parse_pipe_3_get_and_send_tensors_sync_without_ddr()
{
  LOG_NOTICE "Pipe-3: Get and Send Tensors sync (no DDR)"

  $(grep 'ParseKeysTC HBM mode (ms)' $logfile > /dev/null 2>&1)
  if [ $? == 0 ]; then
      grep 'ParseKeysTC HBM mode (ms)' $logfile | \
        awk -F":" 'BEGIN {sum=0; count=0;} {if($NF<2000) {sum+=$NF; count++;}} END \
        {printf "--|ParseKeysTC(filter>2000ms): avg=%0.1f\n", sum/count}'
  fi
}

main()
{
  validate_options $@
  check_spdlog_level

  echo "+----------------------------------------------------------------+"
  echo "+                         Profile Result                         +"
  echo "+----------------------------------------------------------------+"

  parse_pipe_1_data_parser
  parse_pipe_2_key_process

  $(grep 'DDR mode' $logfile > /dev/null 2>&1)
  if [ $? -eq 0 ]; then
      parse_pipe_3_get_and_send_tensors_with_ddr
  else
    $(grep 'ParseKeysTC HBM mode (ms)' $logfile > /dev/null 2>&1)
    if [ $? -eq 0 ]; then
      parse_pipe_3_get_and_send_tensors_sync_without_ddr
    else
      parse_pipe_3_get_tensors_async_no_ddr
      parse_pipe_4_send_tensors_async_no_ddr
    fi
  fi
}

main $@
#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd.
# Description MxRec性能分析脚本 V1.0
set -e

file="$1" #请输入spdlog文件

calculate_average() {
  awk '{
    sum += $1;
    count++
  } END {
    average = sum / count;
    print average
  }'
}
perf() {
  echo "read batch cost"
  cat ${file} | grep 'read batch cost'|grep -v timeout|tail -n 20| awk 'NR%2==1'
  echo "===================================="
  echo "key process cost"
  cat ${file} | grep 'key process cost'|tail
  avg=$(cat ${file} | grep -Po '(?<=key process cost:)[^,:]+(?=,)'|tail -n +20 |calculate_average)
  echo "Average: $avg"
  echo "===================================="
  echo "分析host和device流水，当 host key process 提前训练step时，host性能不为瓶颈"
  echo "按输入训练step打印标志，（默认为step） Enter打开分析，按q退出"
  read step
  step="${step:-step}"
  cat ${file} | grep -P "key process cost|${step}"|tail -n100|less
}
echo -e "\e[45m\e[1m        =========MxRec分析脚本 V1.0=========           \e[0m"
echo

stuck_check() {
  echo -e "\e[106m--------卡住、getnext超时问题定位----------\e[0m"
  echo -n "超时通道为："
  cat ${file} | grep -Po "aicpu_getnext.*GetNext"
  echo
  echo "检查每张卡发送lookup数量："
  for i in {0..7}
  do
      line=$(cat ${file} | grep -P "send"|grep "h2d"|grep "1,${i}"|wc -l)
      echo -n "$line "
  done
  echo
  echo "检查每张卡发送h2d数量是否相同："
  for i in {0..7}
  do
      line=$(cat ${file} | grep "send"|grep "h2d"|grep "1,${i}"|wc -l)
      echo -n "$line "
  done
  echo
  echo "检查每张卡接收数量是否相同："
  for i in {0..7}
  do
      line=$(cat ${file} | grep "r recv"|grep "1,${i}"|wc -l)
      echo -n "$line "
  done
  echo
  echo "每张卡最后接收batch为："
  cat ${file}|grep "trans emb"|grep "info"|tail
}

hot_check() {
  # 查看hot emb去重率
  echo "表名及去重率（去重后/去重前）为：（应该要小于0.4）"
  cat op_summary_*.csv |grep gather_for_restore_vector |awk -F "," '{print $6,$14,$15}'|sed 's/"//g'|sed 's/ [0-9]*;/\//'
}

perf

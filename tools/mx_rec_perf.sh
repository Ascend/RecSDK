#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd.
# Description MxRec性能分析脚本 V1.0

file="$1" #请输入spdlog文件

calculate_average() {
  awk '{
    sum += $1;
    count++
  } END {
    avg = sum / count;
    print avg
  }'
}
echo "        =========MxRec性能分析脚本 V1.0=========           "
echo "read batch cost"
cat ${file} | grep 'read batch cost'|tail -n 20| awk 'NR%2==1'
echo "===================================="
echo "key process cost"
cat ${file} | grep 'key process cost'|tail
avg=`cat ${file} | grep -Po '(?<=key process cost:)[^,:]+(?=ms)'|tail -n +20 |calculate_average`
real_avg=$(echo "$avg" | awk '{ printf "%.2f", $0/6 }')
echo "Average: $real_avg(single thread avg is $avg)"
echo "===================================="
echo "分析host和device流水，当 host key process 提前训练step时，host性能不为瓶颈"
echo "按输入训练step打印标志，（默认为step） Enter打开分析，按q退出"
read step
step="${step:-step}"
cat ${file} | grep -P "key process cost|${step}"|tail -n100|less

exit

# getnext超时问题定位
echo -n "超时通道为："
cat ${file} | grep -Po "aicpu_getnext.*GetNext"
echo - "检查是否发送, 发送数量为注意8卡 "
cat ${file} | grep -P "send" |grep all2all |wc -l
cat ${file} | grep -P "send"|grep h2d|wc -l

echo -n "检查数据读取, 读取batch数量为 "
cat ${file} | grep 'read batch cost'|wc -l
cat ${file} | grep 'read batch cost'|tail



# 查看hot emb去重率
echo "表名及去重率（去重后/去重前）为：（应该要小于0.4）"
cat op_summary_*.csv |grep gather_for_restore_vector |awk -F "," '{print $6,$14,$15}'|sed 's/"//g'|sed 's/ [0-9]*;/\//'

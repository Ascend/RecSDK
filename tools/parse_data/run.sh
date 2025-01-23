#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
# Description: performace analysis tool
# Author: MindSDK
# Create: 2023
# History: NA

for i in {0..7}
do
  nohup python3 data_parser.py $i > rank_$i.log 2>&1 &
done
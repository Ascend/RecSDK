#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
# Description: performace analysis tool
# Author: MindX SDK
# Create: 2023
# History: NA

# cpu with high-performance
cpupower frequency-set -g performance
cat /proc/cpuinfo|grep MHz

# clear cache
echo 3 > /proc/sys/vm/drop_caches
free -h

# swap off
swapoff -a

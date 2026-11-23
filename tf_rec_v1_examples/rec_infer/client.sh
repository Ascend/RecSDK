#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# Description: startup client

source /usr/local/Ascend/ascend-toolkit/set_env.sh
unset http_proxy
unset https_proxy
python3 client.py $1
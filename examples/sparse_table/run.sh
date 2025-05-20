#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# sparse_table_launcher.sh - 分布式训练启动脚本
# 用法: ./sparse_table_launcher.sh [nproc_per_node]

set -e  # 遇到错误立即退出

# 默认参数设置
DEFAULT_NPROC=4
SCRIPT="sparse_table.py"

# 参数解析
if [ $# -gt 0 ]; then
    NPROC_PER_NODE=$1
    # 验证参数是否为数字
    if ! [[ $NPROC_PER_NODE =~ ^[0-9]+$ ]]; then
        echo "错误: nproc_per_node 必须是正整数" >&2
        exit 1
    fi
else
    echo "未指定nproc_per_node，使用默认值 $DEFAULT_NPROC"
    NPROC_PER_NODE=$DEFAULT_NPROC
fi

# 检查torchrun是否可用
if ! command -v torchrun &> /dev/null; then
    echo "错误: torchrun 未找到，请确保PyTorch已正确安装" >&2
    exit 1
fi

# 检查脚本是否存在
if [ ! -f "$SCRIPT" ]; then
    echo "错误: 训练脚本 $SCRIPT 不存在" >&2
    exit 1
fi


# 核心启动命令
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    $SCRIPT

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "训练成功完成"
else
    echo "训练异常终止" >&2
    exit 1
fi

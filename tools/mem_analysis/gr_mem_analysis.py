#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
************************************************************
driver mem = free + reserved

allocated = OS + CANN + Driver + GE + PTA
(Currently, Driver consume 3GB)

PTA = fragmentation + Multi-stream overhead + allocated

allocated = static_mem + active_mem + worksapce

In a model,
    Optimizer: param_fp32, momentum, variance.  All are FP32
    Model:  model_param. Often is bf16/fp16

In specific module(not precisely),
    Linear: B * S * (C_in + C_out)
    Conv:   B * C_in * H_in * W_in + B * C_out * H_out * W_out
    LayerNorm: B * S * H
    Residual Connection: B * S * H

************************************************************

This code will give a demo about memory usage of GR model.
"""
import argparse
from dataclasses import dataclass
import logging


def setup_console_logger(
        logger_name: str = "app",
        log_level: int = logging.INFO,
        show_timestamp: bool = True
) -> logging.Logger:
    """
    配置控制台日志记录器

    参数:
        logger_name: 记录器名称
        log_level: 日志级别 (logging.DEBUG/INFO/WARNING/ERROR/CRITICAL)
        show_timestamp: 是否显示时间戳
    """
    # 创建记录器实例
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # 避免重复添加处理器
    if logger.hasHandlers():
        return logger

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 配置日志格式
    log_format = "%(levelname)s"
    if show_timestamp:
        log_format = "%(asctime)s - " + log_format
    log_format += " - %(message)s"

    formatter = logging.Formatter(
        fmt=log_format,
        datefmt="%H:%M:%S" if show_timestamp else None
    )
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    return logger


GB = 1024 ** 3
B = 1000000000

bf16 = 2
fp32 = 4


@dataclass
class ModelConfig:
    """模型训练和稀疏表的配置参数"""
    tp: int = 2             # 张量并行度
    dp: int = 4             # 数据并行度
    batch_size: int = 1     # 批处理大小
    num_of_item: int = 1024 # 物品数量
    embedding_dim: int = 128 # 嵌入维度大小
    hstu_layer_num: int = 25 # Transformer层数
    seq_length: int = 2048  # 序列长度
    head_num: int = 4       # 注意力头数量
    hidden_size: int = 256  # 隐藏层维度大小


def mem_analysis(model_config, model_param_size):
    tp = model_config.tp
    dp = model_config.dp
    batch_size = model_config.batch_size
    num_of_item = model_config.num_of_item
    embedding_dim = model_config.embedding_dim
    hstu_layer_num = model_config.hstu_layer_num
    seq_length = model_config.seq_length
    head_num = model_config.head_num
    hidden_size = model_config.hidden_size
    # sparse embedding
    sparse_emb = fp32 * num_of_item * embedding_dim * 3 / GB / tp

    # workspace & activation
    max_workspace = 0.8
    hidden_state = hstu_layer_num * ((10 * batch_size * bf16 * head_num * hidden_size * seq_length) + (
            batch_size * bf16 * head_num * hidden_size * seq_length)) / GB
    active_memory = hidden_state + max_workspace

    # optimizer
    m, v = fp32 * model_param_size, fp32 * model_param_size
    fp32_param = fp32 * model_param_size
    grad_data = fp32 * model_param_size
    optimizer = grad_data + (m + v) + fp32_param

    # model
    param_data = bf16 * model_param_size
    static_memory = param_data + optimizer + sparse_emb
    total_model_memory = static_memory + active_memory
    torch_reserved_memory = 2
    ge_reserved_memory = 2
    total_reserved_memory = total_model_memory + torch_reserved_memory + ge_reserved_memory
    return active_memory, static_memory, total_model_memory, total_reserved_memory


if __name__ == '__main__':
    # 初始化参数解析器
    parser = argparse.ArgumentParser(description='模型内存占用计算器')

    # 添加并行配置参数
    parser.add_argument('--tp', type=int, default=2, help='张量并行度')
    parser.add_argument('--dp', type=int, default=4, help='数据并行度')

    # 添加训练配置参数
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')

    # 添加稀疏嵌入配置参数
    parser.add_argument('--num_of_item', type=int, default=1024, help='物品数量')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding维度')

    # 添加模型配置参数
    parser.add_argument('--hstu_layer_num', type=int, default=25, help='hstu层数')
    parser.add_argument('--seq_length', type=int, default=2048, help='序列长度')
    parser.add_argument('--head_num', type=int, default=4, help='注意力头数')
    parser.add_argument('--hidden_size', type=int, default=256, help='隐藏层大小')
    parser.add_argument('--model_size', type=float, default=6.71,
                        help='模型参数数量(以十亿为单位)，不指定则使用公式计算')

    args = parser.parse_args()
    config = ModelConfig(
        tp=args.tp,
        dp=args.dp,
        batch_size=args.batch_size,
        num_of_item=args.num_of_item,
        embedding_dim=args.embedding_dim,
        hstu_layer_num=args.hstu_layer_num,
        seq_length=args.seq_length,
        head_num=args.head_num,
        hidden_size=args.hidden_size,
    )
    if args.model_size:
        model_size = args.model_size * B / GB
    else:
        model_size = (5 * config.embedding_dim * config.embedding_dim * config.hstu_layer_num) / GB
    active_mem, static_mem, total_model_mem, total_reserved = mem_analysis(config, model_size)
    logger = setup_console_logger(
        logger_name="memory_analysis",
        log_level=logging.INFO
    )
    logger.info("active mem: %.2f GB" % active_mem)
    logger.info("static mem: %.2f GB" % static_mem)
    logger.info("total model mem: %.2f GB" % total_model_mem)
    logger.info("reserved memory: %.2f GB" % total_reserved)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import time
import logging

import pandas as pd
import torch

import dynamic_emb_extensions as demb


logging.basicConfig(level=logging.NOTSET)  # 指定日志文件路径等


class OptimizerParams:
    '''优化器参数
        lr: 学习率
        beta1: 一阶衰减系数
        beta2: 二阶衰减系数
        eps: 小常数，防止除零
        weight_decay: 权重衰减系数
    '''
    def __init__(self, lr, beta1, beta2, eps, weight_decay):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay


def get_dim_pointers_optimized(x_2d):
    """利用内存布局特性高效计算首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    # 取出二维张量的维度
    n, m = x_2d.size()
    # 计算每个元素的字节大小
    elem_size = x_2d.element_size()
    # 计算每行的字节偏移量
    row_stride = m * elem_size
    # 计算指针数组
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


def test_dynamic_emb_AdamW_with_pointer(device, batch_size, embedding_dim, optimizer_params, iter_num_max):
    """测试动态嵌入 AdamW 更新算子"""
    # 从optimizer_params中提取参数
    lr = optimizer_params.lr
    beta1 = optimizer_params.beta1
    beta2 = optimizer_params.beta2
    eps = optimizer_params.eps
    weight_decay = optimizer_params.weight_decay

    torch.npu.set_device(device)
    # 生成随机参数
    params = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f'npu:{device}')
    m = torch.zeros_like(params, dtype=torch.float32)
    v = torch.zeros_like(params, dtype=torch.float32)
    # 合并参数和优化器状态
    values = torch.cat([params, m, v], dim=1)
    torch.npu.synchronize(device)

    val_pointers = get_dim_pointers_optimized(values)

    elapsed_time = []
    for iter_num in range(1, iter_num_max + 1):
        grads = torch.randn(batch_size, embedding_dim, dtype=torch.float32, device=f'npu:{device}')
        torch.npu.synchronize(device)
        # 记录开始时间
        start_time = time.perf_counter()
        demb.dynamic_emb_AdamW_with_pointer(grads, val_pointers, demb.DynamicEmbDataType.Float32, embedding_dim * 2, 
                                lr, beta1, beta2, eps, weight_decay, iter_num)
        # 注意同步
        torch.npu.synchronize(device)
        # 记录结束时间
        end_time = time.perf_counter()
        elapsed_time.append((end_time - start_time) * 1e6) # 单位：微秒
    
    # 计算耗时
    time_average = sum(elapsed_time) / iter_num_max # 单位：微秒
    time_max = max(elapsed_time) # 单位：微秒
    time_min = min(elapsed_time) # 单位：微秒
    
    # 返回测试结果数据
    return {
        'Device': device,
        'Batch Size': batch_size,
        'embedding_dim': embedding_dim,
        'grads_num': batch_size * embedding_dim,
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
        'weight_decay': weight_decay,
        'Time Average (us)': time_average,
        'Time Max (us)': time_max,
        'Time Min (us)': time_min,
    }


def main():
    """主函数"""
    cur_device = 0
    iter_num_max = 10
    
    # 预热
    logging.info("warm up")
    warmup_iter_num = 10

    for _ in range(warmup_iter_num):
        result = test_dynamic_emb_AdamW_with_pointer(
        cur_device, 1, 128, OptimizerParams(0.001, 0.9, 0.999, 1e-8, 0.0), iter_num_max)
    
    # 定义测试参数范围
    batch_size_list = [128, 256, 512, 1024, 4096, 8192, 10240, 102400]
    embedding_dim_list = [128, 256, 512, 1024]
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01

    test_cases = []
    for batch_size in batch_size_list:
        for embedding_dim in embedding_dim_list:
            test_cases.append((cur_device, batch_size, embedding_dim,
                               OptimizerParams(lr, beta1, beta2, eps, weight_decay)))
    
    # 收集所有测试结果
    logging.info("start test")
    repeats = 10
    results = []
    for device, batch_size, embedding_dim, optimizer_params in test_cases:
        result = None
        for _ in range(repeats):
            result = test_dynamic_emb_AdamW_with_pointer(device, batch_size, embedding_dim,
                                                         optimizer_params, iter_num_max)
        results.append(result)
    
    # 将结果转换为DataFrame并导出为Excel
    df = pd.DataFrame(results)
    excel_filename = "AdamW_update_time_results.txt"
    df.to_csv(excel_filename, index=False, sep='\t')
    logging.info(f"测试结果已导出到文本文件: {excel_filename}")


if __name__ == "__main__":
    main()

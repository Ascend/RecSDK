#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import threading

import pytest
import torch

import dynamic_emb_extensions as demb


def get_device_timestamp():
    """获取设备时间戳"""
    return demb.device_timestamp()


def test_device_timestamp_basic():
    """测试基本功能"""
    # 获取时间戳
    ts1 = get_device_timestamp()
    assert isinstance(ts1, int)
    # 等待一小段时间
    time.sleep(0.01)
    # 再次获取时间戳
    ts2 = get_device_timestamp()
    # 验证时间戳递增
    assert ts2 > ts1, "时间戳应该递增"
    

def test_device_timestamp_multiple_calls():
    """测试多次调用的一致性"""
    # 连续获取多个时间戳
    timestamps = []
    for _ in range(10):
        ts = get_device_timestamp()
        timestamps.append(ts)
        time.sleep(0.001)  # 等待1ms
    
    # 验证所有时间戳都是递增的
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1], f"时间戳应该递增，但在索引{i}处发现问题"


# 测试多线程并发
def test_device_timestamp_thread_safety():
    """测试多线程并发调用，验证结果的有效性和唯一性"""
    
    # 结果存储为 (thread_id, timestamp) 对
    results = [] 
    
    def get_timestamp(thread_id):
        # 确保每个线程调用两次，以验证递增
        ts1 = get_device_timestamp()
        time.sleep(0.001) # 等待一小段时间
        ts2 = get_device_timestamp()
        
        # 验证每个线程内部的调用是递增的
        assert ts2 > ts1, f"线程{thread_id}内部时间戳ts2({ts2})没有递增于ts1({ts1})"
        
        # 将结果添加到共享列表, Python list.append是线程安全的
        results.append((thread_id, ts1))
        results.append((thread_id, ts2))
    
    # 创建多个线程并发获取时间戳
    num_threads = 100
    threads = []
    
    for i in range(num_threads):
        # 传递线程ID
        t = threading.Thread(target=get_timestamp, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # --- 并发结果验证 ---
    
    # 1. 验证数量
    assert len(results) == num_threads * 2, "获取的时间戳数量不正确"
    
    # 2. 验证有效性：所有时间戳都应该是正整数且在合理范围内。
    all_timestamps = [ts for _, ts in results]
    for ts in all_timestamps:
        assert isinstance(ts, int) and ts > 0, f"时间戳无效: {ts}"
        
    # 3. 验证唯一性：在理想情况下，所有时间戳都应该是唯一的。
    unique_timestamps = set(all_timestamps)
    assert len(unique_timestamps) == len(all_timestamps), "存在重复时间戳，可能因为并发冲突或时间太近"

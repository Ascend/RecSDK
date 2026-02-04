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

import torch
import pandas as pd

import dynamic_emb_extensions as demb


logging.basicConfig(level=logging.NOTSET)  # 指定日志文件路径等


class DynamicEmbTableOptions:
    def __init__(self, key_type=demb.DynamicEmbDataType.Int64,
                 value_type=demb.DynamicEmbDataType.Float32,
                 evict_strategy=demb.EvictStrategy.kLru,
                 dim=128,
                 init_capacity=1024,
                 max_capacity=2048,
                 max_hbm_for_vectors=0,
                 max_bucket_size=128,
                 max_load_factor=0.5,
                 block_size=128,
                 io_block_size=1024,
                 device_id=0,
                 io_by_cpu=False,
                 use_constant_memory=False,
                 reserved_key_start_bit=0,
                 num_of_buckets_per_alloc=1,
                 initializer_args=demb.InitializerArgs(),
                 safe_check_mode=demb.SafeCheckMode.IGNORE,
                 optimizer_type=demb.OptimizerType.Null):
        self.key_type = key_type
        self.value_type = value_type
        self.evict_strategy = evict_strategy
        self.dim = dim
        self.init_capacity = init_capacity
        self.max_capacity = max_capacity
        self.max_hbm_for_vectors = max_hbm_for_vectors
        self.max_bucket_size = max_bucket_size
        self.max_load_factor = max_load_factor
        self.block_size = block_size
        self.io_block_size = io_block_size
        self.device_id = device_id
        self.io_by_cpu = io_by_cpu
        self.use_constant_memory = use_constant_memory
        self.reserved_key_start_bit = reserved_key_start_bit
        self.num_of_buckets_per_alloc = num_of_buckets_per_alloc
        self.initializer_args = initializer_args
        self.safe_check_mode = safe_check_mode
        self.optimizer_type = optimizer_type


def run_performance_test(table_options, repeats=10):
    start_time = time.perf_counter()
    for _ in range(repeats):
        table = demb.DynamicEmbTable(
        table_options.key_type,
        table_options.value_type,
        table_options.evict_strategy,
        table_options.dim,
        table_options.init_capacity,
        table_options.max_capacity,
        table_options.max_hbm_for_vectors,
        table_options.max_bucket_size,
        table_options.max_load_factor,
        table_options.block_size,
        table_options.io_block_size,
        table_options.device_id,
        table_options.io_by_cpu,
        table_options.use_constant_memory,
        table_options.reserved_key_start_bit,
        table_options.num_of_buckets_per_alloc,
        table_options.initializer_args,
        table_options.safe_check_mode,
        table_options.optimizer_type,
        )
        torch.npu.synchronize()
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000000 / repeats # 平均消耗时间，us


def convert_dict_to_readable(options_dict):
    readable_options = {}
    for k, v in options_dict.items():
        if isinstance(v, demb.InitializerArgs):
            readable_options[k] = (
                "InitializerArgs:()"
                )
        elif isinstance(v, (int, float, str, bool)):
            readable_options[k] = v
        elif isinstance(v, demb.DynamicEmbDataType):
            readable_options[k] = v.name
        elif isinstance(v, demb.EvictStrategy):
            readable_options[k] = v.name
        elif isinstance(v, demb.OptimizerType):
            readable_options[k] = v.name
        elif isinstance(v, demb.SafeCheckMode):
            readable_options[k] = v.name
        else:
            readable_options[k] = str(v)
    return readable_options


def main():
    # set device
    cur_device = 0
    torch.npu.set_device(cur_device)
    
    key_type_list = [demb.DynamicEmbDataType.Int64, demb.DynamicEmbDataType.UInt64]
    value_type = demb.DynamicEmbDataType.Float32
    evict_strategy_list = [demb.EvictStrategy.kLru, demb.EvictStrategy.kLfu]
    dim = 128
    init_capacity = 1024
    max_capacity = 2048
    max_hbm_for_vectors = 1 * 1024 * 1024 * 1024
    max_bucket_size = 128
    max_load_factor = 0.5
    block_size = 128
    io_block_size = 1024
    device_id = cur_device
    io_by_cpu = False
    use_constant_memory = False
    reserved_key_start_bit = 0
    num_of_buckets_per_alloc = 1
    initializer_args = demb.InitializerArgs()
    safe_check_mode_list = [demb.SafeCheckMode.IGNORE, demb.SafeCheckMode.ERROR]
    optimizer_type_list = [demb.OptimizerType.Null, demb.OptimizerType.Adam]

    test_cases = []
    for key_type in key_type_list:
        for evict_strategy in evict_strategy_list:
            for safe_check_mode in safe_check_mode_list:
                for optimizer_type in optimizer_type_list:
                    test_cases.append(DynamicEmbTableOptions(
                                        key_type, value_type, evict_strategy, dim, init_capacity,
                                        max_capacity, max_hbm_for_vectors, max_bucket_size,
                                        max_load_factor, block_size, io_block_size, device_id,
                                        io_by_cpu, use_constant_memory, reserved_key_start_bit,
                                        num_of_buckets_per_alloc, initializer_args, safe_check_mode, optimizer_type))
    

    # warmup
    logging.info("warmup...")
    t = None
    for table_options in test_cases:
        t = run_performance_test(table_options)
    logging.info(f"warmup time: {t:.2f} us")

    # test
    logging.info("test...")
    result_list = []
    for table_options in test_cases:
        t_us = run_performance_test(table_options)

        readable_options = convert_dict_to_readable(table_options.__dict__)
        result_list.append((readable_options, t_us))
        logging.info(f"{readable_options}, time: {t_us:.2f} us")

    logging.info("-" * 50)
    logging.info(f"avg time: {sum([t_us for _, t_us in result_list]) / len(result_list):.2f} us")
    logging.info(f"max time: {max([t_us for _, t_us in result_list]):.2f} us")
    logging.info(f"min time: {min([t_us for _, t_us in result_list]):.2f} us")
    
    # 将结果转换为可读性更好的格式，然后保存为csv
    df = pd.DataFrame([
        {
            "key_type": table_options["key_type"],
            "value_type": table_options["value_type"],
            "evict_strategy": table_options["evict_strategy"],
            "dim": table_options["dim"],
            "init_capacity": table_options["init_capacity"],
            "max_capacity": table_options["max_capacity"],
            "max_hbm_for_vectors": table_options["max_hbm_for_vectors"],
            "max_bucket_size": table_options["max_bucket_size"],
            "max_load_factor": table_options["max_load_factor"],
            "block_size": table_options["block_size"],
            "io_block_size": table_options["io_block_size"],
            "device_id": table_options["device_id"],
            "io_by_cpu": table_options["io_by_cpu"],
            "use_constant_memory": table_options["use_constant_memory"],
            "reserved_key_start_bit": table_options["reserved_key_start_bit"],
            "num_of_buckets_per_alloc": table_options["num_of_buckets_per_alloc"],
            "initializer_args": table_options["initializer_args"],
            "safe_check_mode": table_options["safe_check_mode"],
            "optimizer_type": table_options["optimizer_type"],
            "time_us": t_us,
        }
        for table_options, t_us in result_list
    ]
    )
    df.to_csv("perf_create_dynamic_emb_table.txt", index=False, sep="\t")


if __name__ == "__main__":
    main()
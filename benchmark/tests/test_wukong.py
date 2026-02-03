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


import os
import random
import sys
import time

import numpy as np
import torch

wukong_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/wukong-recommendation"))
sys.path.append(wukong_dir)

from model.pytorch import WukongTorch

CPU_ENABLE = os.environ.get("MODEL_CPU_ONLY", "false").upper() == "TRUE"
GPU_ENABLE = torch.cuda.is_available() and not CPU_ENABLE
NPU_ENABLE = not GPU_ENABLE and not CPU_ENABLE

if CPU_ENABLE:
    device = torch.device("cpu")
elif NPU_ENABLE:
    import torch_npu
    device = torch.device("npu")
    torch_npu.npu.matmul.allow_hf32 = True
    torch_npu.npu.conv.allow_hf32 = True
else:
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Running on {device}")

# mock input data
BATCH_SIZE = 1024
NUM_EMBEDDING = 10_000
NUM_CAT_FEATURES = 32
NUM_DENSE_FEATURES = 16

def extend_seed_all(seed=2025):
    os.environ['HCCL_DETERMINISTIC'] = 'True'
    os.environ['CLOSE_MATMUL_K_SHIFT'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
    if NPU_ENABLE:
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
extend_seed_all()

sparse_inputs = torch.multinomial(
    torch.rand((BATCH_SIZE, NUM_EMBEDDING)),
    NUM_CAT_FEATURES,
    replacement=True,
).to(device)
dense_inputs = torch.rand(BATCH_SIZE, NUM_DENSE_FEATURES).to(device)

# takes hyperparameters from the paper
model = WukongTorch(
    num_layers=3,
    num_sparse_emb=NUM_EMBEDDING,
    dim_emb=128,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    num_emb_lcb=16,
    num_emb_fmb=16,
    rank_fmb=24,
    num_hidden_wukong=2,
    dim_hidden_wukong=512,
    num_hidden_head=2,
    dim_hidden_head=512,
    dim_output=1,
).to(device)

inductor_flag = 'inductor' if os.environ.get('MODEL_COMPILE_FLAG', 'False').upper() == 'TRUE' else 'eager'
model_name = os.environ.get("MODEL_NAME", "default_name")

if os.path.exists(f"./save_weights/{model_name}/random_init_weights.pth"):
    state_dict = torch.load(f"./save_weights/{model_name}/random_init_weights.pth", map_location=device)
    model.load_state_dict(state_dict)
else:
    state_dict = model.state_dict()
    for key in state_dict:
        try:
            from torch.distributed._shard.sharded_tensor import ShardedTensor
        except Exception:
            ShardedTensor = ()

        if hasattr(state_dict[key], 'shard') or isinstance(state_dict[key], ShardedTensor):
            print(f"Converting ShardedTensor for key: {key}")
            try:
                state_dict[key] = state_dict[key].local_tensor()
            except:
                state_dict[key] = state_dict[key].to_local()
    os.makedirs(f"./save_weights/{model_name}", exist_ok=True)
    torch.save(state_dict, f"./save_weights/{model_name}/random_init_weights.pth")


output = model(sparse_inputs, dense_inputs)
device_name = str(device).split(":")[0]
os.makedirs(f"../save_results_{device_name}/{model_name}", exist_ok=True)
torch.save(output, f"../save_results_{device_name}/{model_name}/predictions_{model_name}_{inductor_flag}_{0}.pt")

if not CPU_ENABLE:
    steps = int(os.environ.get("MODEL_EPOCH", 10))

    latency_list = []
    with torch.no_grad():
        for _ in range(steps):
            start_time = time.time()
            output = model(sparse_inputs, dense_inputs)
            end_time = time.time()
            latency = end_time - start_time
            latency_list.append(latency)
            print(f"{model_name} time cost: {latency}")

    latency_list = latency_list[2:]
    avg_time = sum(latency_list) / len(latency_list)
    print(f"{model_name} avg time cost: {avg_time}")

    latency_list.sort()
    p90 = 0.0
    p99 = 0.0

    if len(latency_list) > 0:
        p90_index = int(len(latency_list) * 0.9)
        p90_index = min(p90_index, len(latency_list) - 1)
        p90 = round(latency_list[p90_index], 6)

        p99_index = int(len(latency_list) * 0.99)
        p99_index = min(p99_index, len(latency_list) - 1)
        p99 = round(latency_list[p99_index], 6)

    QPS = int(1 / avg_time)
    e2e_result = {
        "Batch_size": BATCH_SIZE,
        "model_name": model_name,
        "QPS": QPS,
        "AVG Latency": avg_time,
        "P99 Latency": p99,
        "P90 Latency": p90
    }

    model_detail_info = device_name + "_" + model_name + "_" + inductor_flag
    eval_str = "performance: " + model_detail_info + "_" + str(e2e_result)
    os.makedirs(f"../save_results_{device_name}/", exist_ok=True)
    with open(f"../save_results_{device_name}/performance_result.txt", "a", encoding="utf-8") as f:
        f.write(eval_str + "\n")

    if (os.environ.get("MODEL_PROFILING_FLAG", "false").upper() == "TRUE"):
        if NPU_ENABLE:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[
                    torch_npu.profiler.ExportType.Text,
                    ],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None
            )

            prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
                ],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                os.path.join("../profiling", model_name),
                worker_name=model_detail_info,
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config)
            prof.start()
        else:
            # GPU profiling configuration
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=1,
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join("../profiling", model_name)),
                record_shapes=False,
                profile_memory=False,
                with_stack=False
            )
            prof.start()
        for _ in range(steps):
            outputs = model(sparse_inputs, dense_inputs)
            prof.step()
        prof.stop()


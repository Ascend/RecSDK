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

import itertools
import sysconfig

import fbgemm_gpu
import pytest
import torch
import torch_npu

from test_comm_utils import (
    ExecuteConfig,
    Scenario,
    VALUES_DATA_TYPES,
    generate_jagged_tensor,
)

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def jagged_2d_to_dense_wrapper(values, offsets, max_lengths, is_mxrec):
    return Jagged2DToDense.apply(values, offsets, max_lengths, is_mxrec)


class Jagged2DToDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, offsets, max_lengths, is_mxrec):
        offsets_tensor = offsets[0] if isinstance(offsets, (list, tuple)) else offsets
        ctx.save_for_backward(offsets_tensor)
        ctx.total_L = values.shape[0]
        ctx.is_mxrec = is_mxrec
        max_sequence_length = max(max_lengths)

        if is_mxrec:
            return torch.ops.mxrec.jagged_2d_to_dense(
                values=values.to(DEVICE),
                offsets=offsets_tensor.to(DEVICE),
                max_sequence_length=max_sequence_length,
            )
        else:
            return torch.ops.fbgemm.jagged_2d_to_dense(
                values=values.to(DEVICE),
                offsets=offsets_tensor.to(DEVICE),
                max_sequence_length=max_sequence_length,
            )

    @staticmethod
    def backward(ctx, grad_output):
        offsets = list(ctx.saved_tensors)
        total_L = ctx.total_L
        is_mxrec = ctx.is_mxrec
        if total_L is None:
            total_L = offsets[0][-1].item()
        if is_mxrec:
            grad_values = torch.ops.mxrec.jagged_to_padded_dense_backward(
                grad_output.to(DEVICE),
                [offsets[0].to(DEVICE)],
                total_L
            )
        else:
            grad_values = torch.ops.fbgemm.jagged_to_padded_dense_backward(
                grad_output.to(DEVICE),
                [offsets[0].to(DEVICE)],
                total_L
            )
        return grad_values, None, None, None, None


def run_case(params):
    dtype = params.get("dtype", torch.float32)
    batch_size = params["batch_size"]
    max_seq_len = params["max_seq_len"]
    num_heads = params.get("num_heads", 2)
    attention_dim = params.get("attention_dim", 16)
    data_types = (dtype, torch.int64)
    jagged_tensor, seq_offsets, _ = generate_jagged_tensor(
        batch_size, max_seq_len, num_heads, attention_dim, data_types=data_types)
    input_flat = jagged_tensor.reshape(jagged_tensor.shape[0], -1)
    offsets_tensor = torch.from_numpy(seq_offsets)

    reference_dense = torch.ops.fbgemm.jagged_2d_to_dense(
        input_flat,
        offsets_tensor,
        max_seq_len
    )
    npu_dense = torch.ops.mxrec.jagged_2d_to_dense(
        input_flat.to(DEVICE),
        offsets_tensor.to(DEVICE),
        max_seq_len
    )
    npu_cpu = npu_dense.cpu()

    assert torch.equal(
        reference_dense.reshape(-1),
        npu_cpu.reshape(-1),
    ), f"NPU结果与FBGEMM CPU结果不匹配\nFBGEMM:\n{reference_dense}\nNPU:\n{npu_cpu}"


test_params = {
    "batch_size": [2, 4],
    "max_seq_len": [128, 256],
    "num_heads": [2, 8],
    "attention_dim": [32],
    "use_list_max_lengths": [False],
    "values_data_type": VALUES_DATA_TYPES,
    "offsets_data_type": [torch.int32, torch.int64],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*test_params.values())
])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_jagged_2d_to_dense(config: ExecuteConfig, is_mxrec: bool):
    """
    测试不规则张量到填充密集张量的转换算子
    测试逻辑:
    1. 生成随机测试数据
    2. 使用FBGEMM的CPU实现计算基准结果
    3. 调用NPU算子计算结果
    4. 对比两者差异(允许1e-4的误差)
    5. 新增: 验证自动求导功能
    """
    batch_size = config.batch_size
    max_seq_len = config.max_seq_len
    num_heads = config.num_heads
    attention_dim = config.attention_dim
    values_data_type = config.values_data_type
    offsets_data_type = config.offsets_data_type

    # 1. 生成测试数据
    data_types = (values_data_type, offsets_data_type)
    jagged_tensor, seq_offsets, total_sequences = generate_jagged_tensor(
        batch_size, max_seq_len, num_heads, attention_dim, data_types)

    # 2. 准备FBGEMM算子输入(需要展平最后两个维度)
    input_flat = jagged_tensor.reshape(total_sequences, num_heads * attention_dim)
    fbgemm_offsets = torch.from_numpy(seq_offsets)

    # ===== 前向传播验证 =====
    # 3. 调用FBGEMM CPU实现
    fbgemm_dense = torch.ops.fbgemm.jagged_2d_to_dense(
        input_flat,
        fbgemm_offsets,
        max_seq_len
    )

    # 4. 调用NPU算子
    if is_mxrec:
        npu_dense = torch.ops.mxrec.jagged_2d_to_dense(
            input_flat.to(DEVICE),
            fbgemm_offsets.to(DEVICE),
            max_seq_len
        )
    else:
        npu_dense = torch.ops.fbgemm.jagged_2d_to_dense(
            input_flat.to(DEVICE),
            fbgemm_offsets.to(DEVICE),
            max_seq_len
        )

    # 5. 前向传播结果比对
    assert torch.equal(
        fbgemm_dense.reshape(-1),
        npu_dense.cpu().reshape(-1),
    ), f"NPU结果与FBGEMM CPU结果不匹配\nFBGEMM:\n{fbgemm_dense}\nNPU:\n{npu_dense.cpu()}"

    # ===== 反向传播验证 =====
    # 6. 准备可训练参数
    input_flat_npu = input_flat.clone().float().to(DEVICE).requires_grad_(True)
    input_flat_npu_py = input_flat.clone().float().to(DEVICE).requires_grad_(True)

    # 7. 计算NPU前向传播
    if is_mxrec:
        npu_dense_for_grad = torch.ops.mxrec.jagged_2d_to_dense(
            input_flat_npu,
            fbgemm_offsets.to(DEVICE),
            max_seq_len
        )
    else:
        npu_dense_for_grad = torch.ops.fbgemm.jagged_2d_to_dense(
            input_flat_npu,
            fbgemm_offsets.to(DEVICE),
            max_seq_len
        )

    # 8. 计算NPU python实现前向传播
    npu_py_dense_for_grad = jagged_2d_to_dense_wrapper(
        input_flat_npu_py,
        [fbgemm_offsets.to(DEVICE)],
        [max_seq_len],
        is_mxrec
    )

    # 9. 生成随机梯度(与输出形状相同)
    grad_output = torch.randn_like(npu_dense_for_grad)

    # 10. NPU反向传播
    npu_dense_for_grad.backward(grad_output.to(DEVICE))
    npu_grad_input = input_flat_npu.grad

    # 11. NPU python反向传播
    npu_py_dense_for_grad.backward(grad_output.to(DEVICE))
    npu_py_grad_input = input_flat_npu_py.grad

    # 12. 梯度比对
    assert torch.equal(
        npu_py_grad_input.cpu(),
        npu_grad_input.cpu(),
    ), f"NPU python梯度与NPU梯度不匹配\nNPU python梯度:\n{npu_py_grad_input.cpu()}\nNPU梯度:\n{npu_grad_input.cpu()}"


SCENARIOS = [
    # 浮点常规场景：覆盖多组 batch / max_seq_len / dtype 组合
    Scenario(
        scenario_id="fp32_padding_values",
        base={"dtype": torch.float32, "num_heads": 2, "attention_dim": 16},
        sweep={
            "batch_size": [2, 4],
            "max_seq_len": [64, 128],
            "dtype": [torch.float32, torch.int64, torch.float16, torch.bfloat16, torch.int32]
        },
    ),
    # 浮点边界场景：小/大 batch 与极端 max_seq_len
    Scenario(
        scenario_id="fp32_edge_cases",
        base={"dtype": torch.float32, "num_heads": 2, "attention_dim": 16},
        sweep={"batch_size": [1, 8, 16], "max_seq_len": [1, 512, 1024]},
    ),
    # 浮点精度场景：极小/极大维度覆盖
    Scenario(
        scenario_id="fp32_precision",
        base={"dtype": torch.float32, "batch_size": 2, "max_seq_len": 32, "num_heads": 2, "attention_dim": 16},
        sweep={},
    ),
    # 浮点维度覆盖：多种 num_heads / attention_dim 组合
    Scenario(
        scenario_id="fp32_dimension_coverage",
        base={"dtype": torch.float32, "batch_size": 2, "max_seq_len": 64},
        sweep={
            "num_heads": [1, 16, 32],
            "attention_dim": [1, 64, 128],
        },
    ),
    # int64 基线：覆盖多种 batch / seq / head 组合
    Scenario(
        scenario_id="int64_baseline",
        base={"dtype": torch.int64},
        sweep={
            "batch_size": [2, 4],
            "max_seq_len": [64, 128],
            "num_heads": [2, 4],
            "attention_dim": [16, 32],
        },
    ),
    # 超大shape测试
    Scenario(
        scenario_id="large_shape_test",
        base={
            "num_heads": 16,
            "attention_dim": 32,
            "max_seq_len": 4096,
            "batch_size": 511,
        },
        sweep={
            "dtype": [torch.int64, torch.float32],
        },
    ),
]

ALL_SCENARIO_CASES = [{"id": sc.scenario_id, "runs": sc.expand()} for sc in SCENARIOS]


@pytest.mark.parametrize("case", ALL_SCENARIO_CASES, ids=lambda c: c["id"])
def test_jagged_2d_to_dense_scenarios(case):
    for idx, run in enumerate(case["runs"]):
        run = dict(run)
        run.setdefault("case_tag", f"{case['id']}_run{idx}")
        run_case(run)


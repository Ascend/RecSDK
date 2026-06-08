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

import logging

import pytest
import torch

import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)

DEVICE = "npu:0"
SMALL_DATA_THRESHOLD = 44 * 1024


@pytest.fixture(autouse=True)
def setup_npu_device():
    torch.npu.set_device(DEVICE)


def to_npu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(DEVICE)


def selected_indices_cpu(flags_cpu: torch.Tensor) -> torch.Tensor:
    if flags_cpu.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device="cpu")
    return torch.nonzero(flags_cpu, as_tuple=True)[0].to(torch.int64)


def torch_select_reference(flags_cpu: torch.Tensor, inputs_cpu: torch.Tensor):
    indices = selected_indices_cpu(flags_cpu)
    if indices.numel() == 0:
        return inputs_cpu.new_empty(0), 0

    # uint64 does not support bool indexing on torch_npu/cpu in some builds
    if inputs_cpu.dtype == torch.uint64:
        expected_outputs = inputs_cpu.to(torch.int64).index_select(0, indices).to(torch.uint64)
    else:
        expected_outputs = inputs_cpu.index_select(0, indices)
    return expected_outputs, expected_outputs.numel()


def torch_select_index_reference(flags_cpu: torch.Tensor, index_dtype: torch.dtype = torch.int64):
    indices = selected_indices_cpu(flags_cpu)
    if index_dtype == torch.uint64:
        indices = indices.to(torch.uint64)
    return indices, indices.numel()


def run_select_op(flags: torch.Tensor, inputs: torch.Tensor):
    assert flags.device.type == "npu"
    assert inputs.device.type == "npu"

    outputs = torch.empty_like(inputs)
    num_selected = torch.empty(1, dtype=torch.int64, device=DEVICE)
    torch.npu.synchronize()
    dynamic_emb_extensions.select(flags, inputs, outputs, num_selected)
    torch.npu.synchronize()
    count = num_selected.item()
    return outputs[:count], num_selected


def run_select_index_op(flags: torch.Tensor, index_dtype: torch.dtype):
    assert flags.device.type == "npu"

    output_indices = torch.empty(flags.numel(), dtype=index_dtype, device=DEVICE)
    num_selected = torch.empty(1, dtype=torch.int64, device=DEVICE)
    torch.npu.synchronize()
    dynamic_emb_extensions.select_index(flags, output_indices, num_selected)
    torch.npu.synchronize()
    count = num_selected.item()
    return output_indices[:count], num_selected


def assert_select_match(actual_outputs, actual_num, expected_outputs, expected_num, **ctx):
    actual_num_val = actual_num.cpu().item()
    assert actual_num_val == expected_num, (
        f"num_selected mismatch: actual={actual_num_val}, expected={expected_num}, {ctx}"
    )

    actual_outputs_cpu = actual_outputs.cpu()
    expected_outputs_cpu = expected_outputs.cpu()
    assert actual_outputs_cpu.shape == expected_outputs_cpu.shape, (
        f"selected shape mismatch: actual={actual_outputs_cpu.shape}, expected={expected_outputs_cpu.shape}, {ctx}"
    )
    assert torch.equal(actual_outputs_cpu, expected_outputs_cpu), (
        f"selected values mismatch, {ctx}\n  actual:   {actual_outputs_cpu}\n  expected: {expected_outputs_cpu}"
    )


def make_inputs(length: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.randint(0, 100000, (length,), generator=generator, dtype=torch.int64, device="cpu")
    if dtype == torch.uint64:
        inputs = inputs.to(torch.uint64)
    elif dtype != torch.int64:
        inputs = inputs.to(dtype)
    return to_npu(inputs)


def make_flags(length: int, true_ratio: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    flags = torch.rand(length, generator=generator, device="cpu") < true_ratio
    return to_npu(flags)


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("length", [0, 1, 10, 100, 1000, 10000, SMALL_DATA_THRESHOLD, SMALL_DATA_THRESHOLD + 1])
@pytest.mark.parametrize("true_ratio", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_select_op_consistency(dtype, length, true_ratio):
    flags = make_flags(length, true_ratio, seed=length * 17 + int(true_ratio * 100))
    inputs = make_inputs(length, dtype, seed=length * 31 + int(true_ratio * 100))

    expected_outputs, expected_num = torch_select_reference(flags.cpu(), inputs.cpu())
    actual_outputs, actual_num = run_select_op(flags, inputs)

    assert_select_match(
        actual_outputs,
        actual_num,
        expected_outputs,
        expected_num,
        dtype=dtype,
        length=length,
        true_ratio=true_ratio,
        op="select",
    )


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("length", [0, 1, 10, 100, 1000, 10000, SMALL_DATA_THRESHOLD, SMALL_DATA_THRESHOLD + 1])
@pytest.mark.parametrize("true_ratio", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_select_index_op_consistency(dtype, length, true_ratio):
    flags = make_flags(length, true_ratio, seed=length * 19 + int(true_ratio * 100))

    expected_indices, expected_num = torch_select_index_reference(flags.cpu(), dtype)
    actual_indices, actual_num = run_select_index_op(flags, dtype)

    assert_select_match(
        actual_indices,
        actual_num,
        expected_indices,
        expected_num,
        dtype=dtype,
        length=length,
        true_ratio=true_ratio,
        op="select_index",
    )

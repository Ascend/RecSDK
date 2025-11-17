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

import logging
import random
import sysconfig
from collections import namedtuple

import torch
import pytest
import fbgemm_gpu

# 加载NPU库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义NPU设备ID
DEVICE_ID = "npu:0"

# 定义反向传播参数命名元组
GradLookupParams = namedtuple(
    "GradLookupParams",
    [
        "devWeights",
        "grad",
        "weightsOffsets",
        "dOffsets",
        "hashSizeCumsum",
        "indices",
        "offsets",
    ],
)


def call_dense_embedding_lookup_grad(params):
    """封装调用dense_embedding_codegen_lookup_function_grad的函数"""
    result = torch.ops.mxrec.dense_embedding_codegen_lookup_function_grad(
        devWeights=params.devWeights,
        grad=params.grad,
        weightsOffsets=params.weightsOffsets,
        dOffsets=params.dOffsets,
        totalD=4,
        maxD=8,
        hashSizeCumsum=params.hashSizeCumsum,
        totalHashSizeBits=2,
        indices=params.indices,
        offsets=params.offsets,
        poolingMode=0,
        indiceWeightsOptional=None,
        featureRequiresGrad=None,
        outputDtypeOptional=0,
        bOffsetOptional=None,
        vbeOutputOffsetsFeatureRankOptional=None,
        vbeBOffsetsRankPerFeatureOptional=None,
        maxB=0,
        maxBFeatureRank=0,
        vbeOutputSize=0,
        mixed_D=False,
    )
    return result


def create_test_data():
    """创建标准测试数据"""
    # 检查NPU是否可用
    if not torch.npu.is_available():
        pytest.skip("NPU is not available, skipping NPU-only tests")

    # 标准输入数据
    dev_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32).to(
        DEVICE_ID
    )
    weights_offsets = torch.tensor([0, 3, 6], dtype=torch.int64).to(DEVICE_ID)
    d_offsets = torch.tensor([0, 2, 4], dtype=torch.int32).to(DEVICE_ID)
    hash_size_cumsum = torch.tensor([0, 2, 4], dtype=torch.int64).to(DEVICE_ID)
    indices = torch.tensor([0, 1, 1, 0], dtype=torch.int64).to(DEVICE_ID)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64).to(DEVICE_ID)
    weights_grad = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=torch.float32
    ).to(DEVICE_ID)

    return GradLookupParams(
        devWeights=dev_weights,
        grad=weights_grad,
        weightsOffsets=weights_offsets,
        dOffsets=d_offsets,
        hashSizeCumsum=hash_size_cumsum,
        indices=indices,
        offsets=offsets,
    )


def test_standard_grad_inputs():
    """测试标准反向输入 - 应该正常运行"""
    params = create_test_data()

    # 调用反向算子
    result = call_dense_embedding_lookup_grad(params)

    # 验证输出形状
    assert result is not None
    assert len(result) > 0
    assert result[0].shape == params.devWeights.shape  # 输出应该与dev_weights形状相同
    logger.info("Standard grad inputs test passed")


def test_empty_tensor_grad_validation():
    """测试反向算子空张量输入校验"""
    params = create_test_data()

    # 测试空的weights_grad
    empty_weights_grad = torch.tensor([], dtype=torch.float32).to(DEVICE_ID)
    empty_params = params._replace(grad=empty_weights_grad)

    with pytest.raises(RuntimeError, match="weightsGrad tensor must be non-empty"):
        call_dense_embedding_lookup_grad(empty_params)

    # 测试空的dev_weights
    empty_dev_weights = torch.tensor([], dtype=torch.float32).to(DEVICE_ID)
    empty_params = params._replace(devWeights=empty_dev_weights)

    with pytest.raises(RuntimeError, match="devWeights tensor must be non-empty"):
        call_dense_embedding_lookup_grad(empty_params)

    logger.info("Empty tensor grad validation test passed")


def test_wrong_dimension_grad_validation():
    """测试反向算子错误维度输入校验"""
    params = create_test_data()

    # 测试错误的weights_grad维度 (应该是2D，但提供1D)
    wrong_dim_weights_grad = params.grad.reshape(-1)
    wrong_params = params._replace(grad=wrong_dim_weights_grad)

    with pytest.raises(RuntimeError, match="weightsGrad must be 2D"):
        call_dense_embedding_lookup_grad(wrong_params)

    # 测试错误的dev_weights维度 (应该是1D，但提供2D)
    wrong_dim_dev_weights = params.devWeights.reshape(2, 3)
    wrong_params = params._replace(devWeights=wrong_dim_dev_weights)

    with pytest.raises(RuntimeError, match="devWeights must be 1D"):
        call_dense_embedding_lookup_grad(wrong_params)

    logger.info("Wrong dimension grad validation test passed")


def test_wrong_dtype_grad_validation():
    """测试反向算子错误数据类型输入校验"""
    params = create_test_data()

    # 测试错误的weights_grad数据类型 (应该是float32，但提供int64)
    wrong_dtype_weights_grad = params.grad.to(torch.int64)
    wrong_params = params._replace(grad=wrong_dtype_weights_grad)

    with pytest.raises(RuntimeError, match="weightsGrad must be float type"):
        call_dense_embedding_lookup_grad(wrong_params)

    # 测试错误的dev_weights数据类型 (应该是float32，但提供int64)
    wrong_dtype_dev_weights = params.devWeights.to(torch.int64)
    wrong_params = params._replace(devWeights=wrong_dtype_dev_weights)

    with pytest.raises(RuntimeError, match="devWeights must be float type"):
        call_dense_embedding_lookup_grad(wrong_params)

    logger.info("Wrong dtype grad validation test passed")


def test_device_mismatch_grad_validation():
    """测试反向算子设备不匹配校验"""
    params = create_test_data()

    # 测试设备不匹配 (CPU tensors with NPU tensors)
    cpu_dev_weights = params.devWeights.cpu()
    cpu_weights_grad = params.grad.cpu()
    cpu_params = params._replace(devWeights=cpu_dev_weights, grad=cpu_weights_grad)

    with pytest.raises(RuntimeError, match="devWeights tensor must be on NPU device"):
        call_dense_embedding_lookup_grad(cpu_params)

    logger.info("Device mismatch grad validation test passed")


def test_offset_indices_mismatch_grad_validation():
    """测试反向算子offsets和indices大小不匹配校验"""
    params = create_test_data()

    # 修改offsets，使其最后一个元素与indices大小不匹配
    wrong_offsets = torch.tensor([0, 1, 3], dtype=torch.int64).to(DEVICE_ID)
    wrong_params = params._replace(offsets=wrong_offsets)

    with pytest.raises(
        RuntimeError, match="offsets last element must match indices size"
    ):
        call_dense_embedding_lookup_grad(wrong_params)

    logger.info("Offset-indices mismatch grad validation test passed")


# 新增测试用例：校验maxD是否为8的倍数
def test_maxd_alignment_validation():
    """
    Test maxD alignment validation - must be a multiple of ALIGNMENT_SIZE (8)
    """
    ALIGNMENT_SIZE = 8
    params = create_test_data()
    # 非法的maxD值 (非8的倍数)
    invalid_maxD = ALIGNMENT_SIZE + 1
    with pytest.raises(RuntimeError, match="maxD must be a multiple of 8"):
        torch.ops.mxrec.dense_embedding_codegen_lookup_function_grad(
            devWeights=params.devWeights,
            grad=params.grad,
            weightsOffsets=params.weightsOffsets,
            dOffsets=params.dOffsets,
            totalD=4,
            maxD=invalid_maxD,
            hashSizeCumsum=params.hashSizeCumsum,
            totalHashSizeBits=2,
            indices=params.indices,
            offsets=params.offsets,
            poolingMode=0,
            indiceWeightsOptional=None,
            featureRequiresGrad=None,
            outputDtypeOptional=0,
            bOffsetOptional=None,
            vbeOutputOffsetsFeatureRankOptional=None,
            vbeBOffsetsRankPerFeatureOptional=None,
            maxB=0,
            maxBFeatureRank=0,
            vbeOutputSize=0,
            mixed_D=False,
        )

    logger.info("MaxD alignment validation with invalid value passed")

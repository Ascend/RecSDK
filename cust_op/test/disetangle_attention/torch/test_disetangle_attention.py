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
import csv
import os
import time
from dataclasses import dataclass

import pytest
import torch
from custom_op import call_custom_op
from gendata import TestArgs, DataArgs, create_test_data
from gloden import golden_disentangle_attention
from verify import compare_result

DEVICE = "npu:0"
torch.npu.set_device(DEVICE)


@dataclass
class CollectInfo:
    batch_size: int
    head_num: int
    seq_len: int
    hea_dim: int
    pos_att_type: str
    atten_weight_compare_res: bool
    atten_prob_compare_res: bool
    atten_output_compare_res: bool
    before_fusion_time_cose: float
    after_fusion_time_cose: float


def write_to_csv(collect_info: CollectInfo):
    # 定义 CSV 文件名和表头
    csv_file = "performance_report.csv"
    headers = [
        "batch_size",
        "head_num",
        "seq_len",
        "hea_dim",
        "pos_att_type",
        "atten_weight_compare_res",
        "atten_prob_compare_res",
        "atten_output_compare_res",
        "before_fusion_time_cose(ms)",
        "after_fusion_time_cose(ms)",
    ]

    # 检查文件是否存在，不存在则写入表头
    file_exists = os.path.isfile(csv_file)

    # 以追加模式打开文件
    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(headers)

        # 写入性能数据
        writer.writerow(
            [
                collect_info.batch_size,
                collect_info.head_num,
                collect_info.seq_len,
                collect_info.hea_dim,
                collect_info.pos_att_type,
                collect_info.atten_weight_compare_res,
                collect_info.atten_prob_compare_res,
                collect_info.atten_output_compare_res,
                f"{collect_info.before_fusion_time_cose:.3f}",  # 保留6位小数
                f"{collect_info.after_fusion_time_cose:.3f}",
            ]
        )


def run_test(args: TestArgs, test_cnt: int):
    start_time = time.time()
    torch.npu.synchronize()
    for _ in range(test_cnt):
        golden_atten_outputs, golden_atten_probs, golden_atten_weights = golden_disentangle_attention(args)
    torch.npu.synchronize()
    end_time = time.time()
    golden_time_cose = (end_time - start_time) * 1000 / test_cnt

    start_time = time.time()
    torch.npu.synchronize()
    for _ in range(test_cnt):
        result_atten_outputs, result_atten_probs, result_atten_weights = call_custom_op(args)
    torch.npu.synchronize()
    end_time = time.time()
    op_time_cose = (end_time - start_time) * 1000 / test_cnt

    res1, res2, res3 = compare_result(
        {
            "atten_weights": golden_atten_weights,
            "atten_probs": golden_atten_probs,
            "atten_outputs": golden_atten_outputs,
        },
        {
            "atten_weights": result_atten_weights,
            "atten_probs": result_atten_probs,
            "atten_outputs": result_atten_outputs,
        },
    )

    return res1, res2, res3, golden_time_cose, op_time_cose


@pytest.mark.parametrize("b", [1, 11, 48])
@pytest.mark.parametrize("n", [1, 12, 23])
@pytest.mark.parametrize("s", [256])
@pytest.mark.parametrize("d", [16, 32, 64])
@pytest.mark.parametrize("pos_att_type", ["c2p", "p2c", "c2p|p2c"])
def test_main(b, n, s, d, pos_att_type):
    args = create_test_data(DataArgs(b, n, s, d, -1, 512, pos_att_type), DEVICE)

    warm_up_cnt: int = 10
    # warm up
    _, _, _, _, _ = run_test(args, warm_up_cnt)

    test_cnt: int = 100
    # profiling test
    (
        weight_compare_res,
        prob_comapre_res,
        output_compare_res,
        golden_time_cose,
        op_time_cose,
    ) = run_test(args, test_cnt)

    collect_info = CollectInfo(
        b,
        n,
        s,
        d,
        pos_att_type,
        weight_compare_res,
        prob_comapre_res,
        output_compare_res,
        golden_time_cose,
        op_time_cose,
    )

    write_to_csv(collect_info)

    assert weight_compare_res
    assert prob_comapre_res
    assert output_compare_res


# ruff: off
def test_disentangle_attention_exceptions():
    """测试 disentangle_attention 算子的异常校验"""
    # pylint: disable=unused-variable,line-too-long
    # 默认合法参数规格
    b, n, s, d = 2, 4, 256, 32
    q = torch.randn(b, n, s, d, dtype=torch.float16, device=DEVICE)
    k = torch.randn(b, n, s, d, dtype=torch.float16, device=DEVICE)
    v = torch.randn(b, n, s, d, dtype=torch.float16, device=DEVICE)
    pk = torch.randn(2 * s, n, d, dtype=torch.float16, device=DEVICE)
    pq = torch.randn(2 * s, n, d, dtype=torch.float16, device=DEVICE)
    rel_pos = torch.randint(0, 10, (s, s), dtype=torch.int64, device=DEVICE)
    mask = torch.randn(b, 1, s, s, dtype=torch.float16, device=DEVICE)

    # 1. 测试 dtype 校验（例如 query_layer 传入 float32）
    q_err = q.to(torch.float32)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.disentangle_attention(q_err, k, v, pk, pq, rel_pos, mask, "c2p", 1.0)
    assert "float16 query_layer tensor expected" in str(ctx.value)

    # 2. 测试 relative_pos dtype 校验（例如传入 int32）
    rel_pos_err = rel_pos.to(torch.int32)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.disentangle_attention(q, k, v, pk, pq, rel_pos_err, mask, "c2p", 1.0)
    assert "int64 relative_pos tensor expected" in str(ctx.value)

    # 3. 测试 query_layer 维度非 4D (例如 3D)
    q_dim_err = torch.randn(b, s, d, dtype=torch.float16, device=DEVICE)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.disentangle_attention(q_dim_err, k, v, pk, pq, rel_pos, mask, "c2p", 1.0)
    assert "query_layer expect 4 dim" in str(ctx.value)

    # 4. 测试 value_layer 与 query_layer 尺寸不一致
    v_shape_err = torch.randn(b, n, s, d * 2, dtype=torch.float16, device=DEVICE)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.disentangle_attention(q, k, v_shape_err, pk, pq, rel_pos, mask, "c2p", 1.0)
    assert "query_layer format must equal value_layer format" in str(ctx.value)

    # 5. 测试 seq 长度非 256 (例如 128)
    q_seq_err = torch.randn(b, n, 128, d, dtype=torch.float16, device=DEVICE)
    pk_seq_err = torch.randn(256, n, d, dtype=torch.float16, device=DEVICE)
    pq_seq_err = torch.randn(256, n, d, dtype=torch.float16, device=DEVICE)
    rel_pos_seq_err = torch.randint(0, 10, (128, 128), dtype=torch.int64, device=DEVICE)
    mask_seq_err = torch.randn(b, 1, 128, 128, dtype=torch.float16, device=DEVICE)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.disentangle_attention(
            q_seq_err,
            q_seq_err,
            q_seq_err,
            pk_seq_err,
            pq_seq_err,
            rel_pos_seq_err,
            mask_seq_err,
            "c2p",
            1.0,
        )
    assert "current seq only support 256" in str(ctx.value)


# ruff: on

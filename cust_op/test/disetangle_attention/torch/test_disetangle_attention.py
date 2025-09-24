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
import csv
import os
import time
from dataclasses import dataclass

import pytest
import torch
import torch_npu
from custom_op import call_custom_op
from gendata import TestArgs, DataArgs, create_test_data
from gloden import gloden_disetangle_attention
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
        gloden_atten_outputs, gloden_atten_probs, gloden_atten_weights = (
            gloden_disetangle_attention(args)
        )
    torch.npu.synchronize()
    end_time = time.time()
    gloden_time_cose = (end_time - start_time) * 1000 / test_cnt

    start_time = time.time()
    torch.npu.synchronize()
    for _ in range(test_cnt):
        result_atten_outputs, result_atten_probs, result_atten_weights = call_custom_op(
            args
        )
    torch.npu.synchronize()
    end_time = time.time()
    op_time_cose = (end_time - start_time) * 1000 / test_cnt

    res1, res2, res3 = compare_result(
        {
            "atten_weights": gloden_atten_weights,
            "atten_probs": gloden_atten_probs,
            "atten_outputs": gloden_atten_outputs,
        },
        {
            "atten_weights": result_atten_weights,
            "atten_probs": result_atten_probs,
            "atten_outputs": result_atten_outputs,
        },
    )

    return res1, res2, res3, gloden_time_cose, op_time_cose


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
        gloden_time_cose,
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
        gloden_time_cose,
        op_time_cose,
    )

    write_to_csv(collect_info)

    assert weight_compare_res
    assert prob_comapre_res
    assert output_compare_res

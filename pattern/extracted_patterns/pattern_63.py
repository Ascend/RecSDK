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

import torch

from utils.logger import default_logger


class PatternModel(torch.nn.Module):
    def __init__(self):
        super(PatternModel, self).__init__()

    def forward(self, input_tensors_1, input_tensors_2, input_tensors_3, mul_tensors, cat_tensors):
        # Step 1: 输出shape 为 (3072,10,16)
        slice_results_1 = [t[:, :, -16:] for t in input_tensors_1]
        addn_result_1 = torch.stack(slice_results_1).sum(dim=0)

        slice_results_2 = [t[:, :, -16:] for t in input_tensors_2]
        addn_result_2 = torch.stack(slice_results_2).sum(dim=0)

        slice_results_3 = [t[:, :, -16:] for t in input_tensors_3]
        addn_result_3 = torch.stack(slice_results_3).sum(dim=0)
        # Step 2: 输出shape 为 (3072,10,48)
        cat_result = torch.cat([addn_result_1, addn_result_2, addn_result_3], dim=-1)
        # Step 3: 输出shape 为 (3072,10,48)

        mul_result = torch.mul(cat_result, mul_tensors)
        # Step 4: 输出shape 为 (3072,10,64)

        final_result = torch.cat([mul_result, cat_tensors], dim=-1)
        return final_result


def main():
    # 示例输入
    input_tensors_1 = [torch.randn([3072, 10, 65]) for _ in range(6)]
    input_tensors_2 = [torch.randn([3072, 10, 65]) for _ in range(6)]
    input_tensors_3 = [torch.randn([3072, 10, 65]) for _ in range(6)]
    mul_tensors = torch.randn([3072, 1, 48])
    cat_tensors = torch.randn([3072, 10, 16])
    model = PatternModel()

    output_tensor = model(input_tensors_1, input_tensors_2, input_tensors_3, mul_tensors, cat_tensors)

    # 打印输出形状
    default_logger.info("Output shape: %s", output_tensor.shape)

if __name__ == "__main__":
    main()
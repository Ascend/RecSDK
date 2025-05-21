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
    def forward(self, input1, input2, input3, input4, input5, input6, input7):
        # Step: Slice each tensor along the last dimension
        slice1 = input1[:, :, :128]     # shape: (128, 50, 128)
        slice2 = input2[:, :, :32]      # shape: (128, 50, 32)
        slice3 = input3[:, :, :48]      # shape: (128, 50, 48)
        slice4 = input4[:, :, :48]      # shape: (128, 50, 48)
        slice5 = input5[:, :, :48]      # shape: (128, 50, 48)
        slice6 = input6[:, :, :48]      # shape: (128, 50, 48)
        slice7 = input7[:, :, :48]      # shape: (128, 50, 48)

        # Step: Concatenate all slices along dim=2
        output_tensor = torch.cat([slice1, slice2, slice3, slice4, slice5, slice6, slice7], dim=2)

        return output_tensor
    

def main():
    # 创建输入张量
    input_shape = (128, 50, 128)
    input1 = torch.randn(input_shape)
    input2 = torch.randn(input_shape)
    input3 = torch.randn(input_shape)
    input4 = torch.randn(input_shape)
    input5 = torch.randn(input_shape)
    input6 = torch.randn(input_shape)
    input7 = torch.randn(input_shape)
    
    model = PatternModel()

    # 运行模型
    output_tensor = model(input1, input2, input3, input4, input5, input6, input7)

    # 打印输出张量的形状 应该输出: torch.Size([128, 50, 400])
    default_logger.info("Output shape: %s", output_tensor.shape)

if  __name__ == "__main__":
    main()
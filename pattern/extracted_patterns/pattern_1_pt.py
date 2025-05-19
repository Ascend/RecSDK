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
import torch.nn as nn

from utils.logger import default_logger
    
class PatternModel(nn.Module):
    def __init__(self):
        super(PatternModel, self).__init__()
        # BatchNormalization 层
        self.batch_norm = nn.BatchNorm2d(256)  # 输入通道数为 256

    def forward(self, input_tensor):
        # Step 1: reshape 为 (128, 192, 1, 256)
        reshaped_tensor = input_tensor.view(128, 192, 1, 256)

        # Step 2: BatchNormalization
        # 注意：PyTorch 的 BatchNorm2d 需要将通道放在第 2 维，因此需要转置
        bn_input = reshaped_tensor.permute(
            0, 3, 1, 2
        )  # 转换为 (batch, channels, height, width)
        batch_norm_tensor = self.batch_norm(bn_input)

        # Step 3: 再次 reshape 为 (128, 192, 256)
        reshaped_bn_tensor = batch_norm_tensor.permute(0, 2, 3, 1).view(128, 192, 256)

        # Step 4: 创建一个常量张量，形状为 (128, 192, 256)
        constant_tensor = torch.ones_like(reshaped_bn_tensor)

        # Step 5: 做逐元素相减操作
        sub_output = reshaped_bn_tensor - constant_tensor

        # 最后，reshape 为 (128, 192, 256)
        final_result = sub_output.view(128, 192, 256)

        return final_result


def main():
    # 示例输入
    input_tensor = torch.randn(128, 192, 256)

    model = PatternModel()

    output_tensor = model(input_tensor)

    # 打印输出形状
    default_logger.info("Output shape: %s", output_tensor.shape)  # 应该输出: torch.Size([128, 192, 256])
    
if __name__ == "__main__":
    main()
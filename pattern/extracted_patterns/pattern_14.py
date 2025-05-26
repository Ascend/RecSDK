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

from pattern.util import perform_test


class PatternModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, start_indices):
        """
        x: 输入张量形状 [batch, row, col]
        输出形状 [batch, num_slices, row, slice_len]
        """
        batch_size, row, col = x.shape
        num_slices = start_indices.size(0)
        slice_length = 128

        # shape -> [num_slices, slice_length]
        indices = start_indices.view(-1, 1) + torch.arange(3)
        # shape -> [batch, num_slices, slice_length]
        indices = indices.expand(batch_size, -1, -1)
        # shape -> [batch, num_slices, row, slice_length]
        indices = indices.unsqueeze(2).expand(-1, -1, row, -1)

        # shape -> [batch, num_slices, row, col]
        x_expanded = x.unsqueeze(1).expand(-1, num_slices, -1, -1)

        x_sliced = x_expanded.gather(dim=3, index=indices)

        return x_sliced


def main():
    batch_size = 256
    input_row = 1000
    input_col = 2048
    slice_num = 4

    torch.manual_seed(2025)
    start_indices = torch.tensor([256, 512, 700, 800])
    input_tensor = torch.randn(batch_size, input_row, input_col)

    input_list = [input_tensor, start_indices]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()

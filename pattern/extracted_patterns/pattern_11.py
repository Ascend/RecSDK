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

    def forward(self, x, indices):
        """
        x:  [batch_size, row, col]
        output: [batch_size, len(indices), col]
        """
        outputs = []
        row = x.size(1)
        for i in range(len(indices)):
            start = indices[i, 0]
            end = indices[i, 1]
            # Replace dynamic slicing with mask operations；original code: sliced = x[:, start:end, :]
            mask = torch.arange(row, device=x.device).ge(start) & torch.arange(row, device=x.device).lt(end)
            masked = x * mask.unsqueeze(0).unsqueeze(-1).expand(x.shape)
            summed = masked.sum(dim=1)
            outputs.append(summed)

        return torch.stack(outputs, dim=1)


def main():
    batch_size = 256
    row = 10
    col = 5

    indices = torch.tensor([[1, 3], [4, 7], [8, 9]])
    torch.manual_seed(2025)
    input_tensor = torch.randint(0, 10, (batch_size, row, col)).float()

    input_list = [input_tensor, indices]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()

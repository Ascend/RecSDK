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

    def forward(self, input0):
        sliced = input0[:, :, 1:1 + 128]
        return sliced.sum(dim=1)


def main():
    y_dim = 256
    x_dim = 64
    r_dim = 34

    in0 = torch.randn(y_dim, x_dim, r_dim)

    input_list = [in0]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()

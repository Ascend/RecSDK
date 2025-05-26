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

    def forward(self, in0, in1):
        add0 = in0 + in1
        out = add0[:, 2:, :]
        return out


def main():
    torch.manual_seed(2025)
    in0 = torch.randn(1, 248, 128, dtype=torch.float16)
    in1 = torch.randn(1, 1, 128, dtype=torch.float16)

    input_list = [in0, in1]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()

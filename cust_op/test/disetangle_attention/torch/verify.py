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

import torch


def compare_percesion(gloden, result, name: str):
    expect = gloden[name].cpu()
    actual = result[name].cpu()

    res = torch.allclose(expect, actual, 1e-3, 1e-3)
    return res


def compare_result(gloden, result):
    res1 = compare_percesion(gloden, result, "atten_weights")
    res2 = compare_percesion(gloden, result, "atten_probs")
    res3 = compare_percesion(gloden, result, "atten_outputs")
    return res1, res2, res3

#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch
from torchrec import KeyedJaggedTensor

from dataset import Batch


def permute_values(kjt: KeyedJaggedTensor, feature_num) -> torch.Tensor:
    keys_nums = feature_num
    values = []
    jt_dict = kjt.to_dict()
    for k in range(keys_nums):
        k = f"feat{k}"
        jt = jt_dict[k]
        values.append(jt)
    values = torch.concat(values, dim=1)
    return values


# ec和ebc查询结果返回数据类型不一样,不同的处理方式
def permute_values_ec(result: Dict, feature_num) -> torch.Tensor:
    keys_nums = feature_num
    values = []
    for k in range(keys_nums):
        k = f"feat{k}"
        jt = result[k].values()
        values.append(jt)
    values = torch.concat(values, dim=1)
    return values


class Model(torch.nn.Module):
    def __init__(self, ebc, feature_num):
        super().__init__()
        self._ebc = ebc
        self.feature_num = feature_num

    @property
    def ebc(self):
        return self._ebc

    def forward(self, batch: Batch):
        result = self._ebc(batch.sparse_features)
        result = permute_values(result, self.feature_num)
        loss = result.sum()
        return loss, result


class ModelEc(torch.nn.Module):
    def __init__(self, ec, feature_num):
        super().__init__()
        self._ec = ec
        self.feature_num = feature_num

    @property
    def ec(self):
        return self._ec

    def forward(self, batch: Batch):
        result = self._ec(batch.sparse_features)
        result = permute_values_ec(result, self.feature_num)
        loss = result.sum()
        return loss, result

#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Type

import torch


def get_embedding_optim_num(optimizer_class: Type[torch.optim.Optimizer]) -> int:
    optim_cls_2_optim_num = {
        torch.optim.Adagrad: 1, 
        torch.optim.Adam: 2,
    }
    return optim_cls_2_optim_num.get(optimizer_class, 0)

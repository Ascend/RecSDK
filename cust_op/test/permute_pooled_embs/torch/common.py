#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
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

from typing import Any

import fbgemm_gpu
import torch
from fbgemm_gpu.permute_pooled_embedding_modules import PermutePooledEmbeddings
from torch import Tensor


INTERN_MODULE = "fbgemm_gpu.permute_pooled_embedding_modules"
FIXED_EXTERN_API = {
    "PermutePooledEmbeddings": {
        "__init__": ["self", "embs_dims", "permute", "device"],
        "__call__": ["self", "pooled_embs"],
    },
}

FWD_COMPAT_MSG = (
    "WARNING: If this test is failing, you are probably trying "
    "to make changes to a module that has been marked external to PyPer torch packages. "
    "This can break forward compatibility of torch packages on training_platform "
    "(see https://fb.workplace.com/groups/pyper/permalink/808155810065803/). "
    "You need to split up your changes as follows:\n"
    "\t1. Edit your diff so it only contains the changes as optional, and not any usage of the"
    " new optional changes.\n"
    "\t2. Edit FIXED_EXTERN_API in this test so your diff passes the test.\n"
    "\t3. Land your diff and wait for the diff to be picked up by the production version of"
    " fbpkg training_platform.\n"
    "\t4. Once step 3. is complete, you can push the rest of your changes that use the new"
    " changes."
)


class PermutePooledEmbeddingsFwdOnly(PermutePooledEmbeddings):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, pooled_embs: torch.Tensor) -> torch.Tensor:
        result = torch.ops.fbgemm.permute_pooled_embs(
            pooled_embs,
            self._offset_dim_list.to(device=pooled_embs.device),
            self._permute.to(device=pooled_embs.device),
            self._inv_offset_dim_list.to(device=pooled_embs.device),
            self._inv_permute.to(device=pooled_embs.device),
        )
        return result


class Net(torch.nn.Module):
    def __init__(self, fwd_only: bool = False) -> None:
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 10, bias=False)
        op_cls = PermutePooledEmbeddingsFwdOnly if fwd_only else PermutePooledEmbeddings
        self.permute_pooled_embeddings: PermutePooledEmbeddings = op_cls(
            [2, 3, 1, 4],
            [3, 0, 2, 1],
        )
        self.fc2 = torch.nn.Linear(10, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.permute_pooled_embeddings(x)
        x = self.fc2(x)
        return x

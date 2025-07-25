#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeVar

import torch

from torchrec.distributed.embedding_types import KJTList
from torchrec.streamable import Pipelineable

In = TypeVar("In", bound=Pipelineable)


def kjt_list_to_device(
    batch: KJTList, device: torch.device, non_blocking: bool = False
) -> In:
    if not isinstance(batch, (KJTList)):
        raise ValueError(f"{type(batch)} must be KJTList")

    result = KJTList(
        [
            f.pin_memory().to(device=device, non_blocking=non_blocking)
            for f in batch.features
        ]
    )
    return result

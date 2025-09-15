#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Type

import torch

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hybrid_torchrec import optim as hybrid_optim
from torchrec import optim as trec_optim


def hybrid_optimizer_type_to_emb_opt_type(
        optimizer_class: Type[torch.optim.Optimizer],
) -> Optional[EmbOptimType]:
    # may need to add special handling for them
    optimizer_type = {
        torch.optim.SGD: EmbOptimType.EXACT_SGD,
        torch.optim.Adagrad: EmbOptimType.EXACT_ADAGRAD,
        torch.optim.Adam: EmbOptimType.ADAM,
        # below are torchrec wrappers over these optims.
        # they accept an **unused kwargs portion, that let us set FBGEMM specific args such as
        # max gradient, etc
        trec_optim.SGD: EmbOptimType.EXACT_SGD,
        trec_optim.Adam: EmbOptimType.ADAM,
        trec_optim.Adagrad: EmbOptimType.EXACT_ADAGRAD,
        hybrid_optim.AccumulateAdagrad: EmbOptimType.EXACT_ADAGRAD,
    }
    if optimizer_class not in optimizer_type:
        raise ValueError(f"Cannot cast {optimizer_class} to an EmbOptimType")
    return optimizer_type[optimizer_class]
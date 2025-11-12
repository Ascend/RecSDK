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

from pathlib import Path

import torch
import torch_npu
from gendata import TestArgs

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(
    str(
        CURR_DIR.parent.parent.parent
        / "cust_op/framework/torch_plugin/torch_library/disetangle_attention/build/libdisentangle_attention.so"
    )
)


def call_custom_op(op_args: TestArgs):
    atten_outpus, atten_probs, atten_weights = torch.ops.mxrec.disentangle_attention(
        op_args.query_layer,
        op_args.key_layer,
        op_args.value_layer,
        op_args.pos_key_layer,
        op_args.pos_query_layer,
        op_args.relative_pos,
        op_args.atten_mask,
        op_args.pos_att_type,
        op_args.score_scale,
    )
    return atten_outpus, atten_probs, atten_weights

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
import torch.nn.functional as F
from gendata import TestArgs


def gloden_disetangle_bias(op_args: TestArgs):
    att_span = op_args.query_layer.size(-2)
    score = None
    if "c2p" in op_args.pos_att_type:
        c2p_att = torch.matmul(
            op_args.query_layer, op_args.pos_key_layer.permute(1, 2, 0)
        )
        c2p_pos = torch.clamp(op_args.relative_pos + att_span - 1, 0, att_span * 2 - 1)
        c2p_pos = c2p_pos.expand([c2p_att.size(0), c2p_att.size(1), -1, -1])
        c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos)

        score = c2p_att

    if "p2c" in op_args.pos_att_type:
        p2c_att = torch.matmul(
            op_args.key_layer, op_args.pos_query_layer.permute(1, 2, 0)
        )
        p2c_pos = torch.clamp(op_args.relative_pos + att_span - 1, 0, att_span * 2 - 1)
        p2c_pos = p2c_pos.expand([p2c_att.size(0), p2c_att.size(1), -1, -1])
        p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos)

        p2c_att = p2c_att.transpose(-1, -2)
        if score is None:
            score = p2c_att
        else:
            score += p2c_att

    if score is None:
        return 0
    return score * op_args.score_scale


def gloden_disetangle_attention(op_args: TestArgs):

    attn_weights = torch.matmul(
        op_args.query_layer * op_args.score_scale, op_args.key_layer.permute(0, 1, 3, 2)
    )

    attn_weights = attn_weights + gloden_disetangle_bias(op_args)

    attn_weights = attn_weights + op_args.atten_mask
    attn_probs = F.softmax(attn_weights, dim=-1)
    attn_outputs = torch.matmul(attn_probs, op_args.value_layer)
    return attn_outputs, attn_probs, attn_weights

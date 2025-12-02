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
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepctr_torch.models.dien import *
from deepctr_torch.layers import DNN, AttentionSequencePoolingLayer


class MyGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True ):
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_w = nn.Parameter(torch.Tensor(input_size, 3 * hidden_size))
        self.weight_u = nn.Parameter(torch.Tensor(hidden_size, 3 * hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in (self.weight_w , self.weight_u):
            nn.init.uniform_(weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
        bias_b, time_t, hate_h = x.size()
        device = x.device
        dtype = x.dtype
        lengths = lengths.to(device)

        gates_x_all = torch.matmul(x, self.weight_w)
        if self.bias is not None:
            gates_x_all = gates_x_all + self.bias

        x_rz_all = gates_x_all[:, :, :2 * hate_h]
        x_n_all = gates_x_all[:, :, 2 * hate_h:]

        h_t = torch.zeros(bias_b, hate_h, device=device, dtype=dtype)
        times_ids = torch.arange(time_t, device = device).unsqueeze(0).expand(bias_b, time_t)
        raw_mask = (times_ids < lengths.unsqueeze(1))
        use_mask = bool((raw_mask == 0).any())
        mask = raw_mask.to(dtype)
        output = []

        for t in range(time_t):
            x_rz_t = x_rz_all[:, t, :]
            x_n_t = x_n_all[:, t, :]

            gates_h = torch.matmul(h_t, self.weight_u)
            h_rz = gates_h[:, :2 * hate_h]
            h_n = gates_h[:, 2 * hate_h:]

            pre_rz = torch.add(x_rz_t, h_rz)
            rz = torch.sigmoid(pre_rz)
            r, z = rz.chunk(2, dim = 1)

            n_input = torch.add(x_n_t, torch.mul(r, h_n))
            h_tilde = torch.tanh(n_input)

            one_minus_z = torch.sub(1.0, z)
            term_new = torch.mul(one_minus_z, h_tilde)
            term_old = torch.mul(z, h_t)
            h_new = torch.add(term_new, term_old)

            if use_mask:
                mask_t = mask[:, t].unsqueeze(1)
                h_t = torch.mul(h_new, mask_t) + torch.mul(h_t.detach(), torch.sub(1.0, mask_t))
            else:
                h_t = h_new
            output.append(h_t.unsqueeze(1))
        return torch.cat(output, dim = 1)


class MyExtractor(nn.Module):
    def __init__(self, input_size: int, use_neg: bool = False, init_std: float = 0.001, device: str = 'cpu'):
        super(MyExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = MyGRU(input_size = input_size, hidden_size = input_size)
        if DNN is not None and use_neg:
            self.auxiliary_net = DNN(input_size * 2, [100, 50, 1], activation = 'sigmoid')
        else:
            self.auxiliary_net = None
        self.to(device)

    def forward(self, keys: torch.Tensor, keys_length: torch.LongTensor,
                neg_keys: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        # keys: [B, T, H], keys_length: [B]
        bias_b, time_t, hate_h = keys.size()
        device = keys.device
        # compute interests: [bias_b, time_t, hate_h]
        interests = self.gru(keys, keys_length)

        aux_loss = torch.zeros((1,), device=device)
        if self.use_neg and (neg_keys is not None) and (self.auxiliary_net is not None):
            # compute auxiliary loss similar to original: states aligned with next-step click/no-click
            # we need states for t=0..T-2 vs clicked item at t+1
            # create mask for valid lengths > 1
            valid_mask = (keys_length > 1)
            if valid_mask.sum() > 0:
                # select valid entries
                sel_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
                states = interests[sel_idx, :-1, :]  # [b_valid, T-1, H]
                click_seq = keys[sel_idx, 1:, :]
                noclick_seq = neg_keys[sel_idx, 1:, :]
                lengths = (keys_length[sel_idx] - 1).clamp(min=0)
                aux_loss = self._cal_auxiliary_loss(states, click_seq, noclick_seq, lengths)
        return interests, aux_loss

    def _cal_auxiliary_loss(self, states: torch.Tensor, click_seq: torch.Tensor,
                            noclick_seq: torch.Tensor, keys_length: torch.LongTensor) -> torch.Tensor:

        # states: [b, T, H], click_seq: [b, T, H], noclick_seq: [b, T, H], keys_length: [b]
        device = states.device
        batch_size, max_seq_length, embedding_size = states.size()
        # mask: [b, T]
        mask = (torch.arange(max_seq_length, device=device).unsqueeze(0).expand(batch_size, max_seq_length)
                < keys_length.unsqueeze(1)).to(states.dtype)

        click_input = torch.cat([states, click_seq], dim=-1)  # [b, T, 2H]
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        emb = embedding_size * 2
        click_p = self.auxiliary_net(
            click_input.view(batch_size * max_seq_length, emb)
        ).view(batch_size, max_seq_length)
        noclick_p = self.auxiliary_net(
            noclick_input.view(batch_size * max_seq_length, emb)
        ).view(batch_size, max_seq_length)

        click_p = click_p[mask > 0].view(-1, 1)
        noclick_p = noclick_p[mask > 0].view(-1, 1)

        if click_p.numel() == 0:
            return torch.zeros((1,), device=device)

        click_target = torch.ones_like(click_p)
        noclick_target = torch.zeros_like(noclick_p)
        loss = F.binary_cross_entropy(
            torch.cat([click_p, noclick_p], dim=0), torch.cat([click_target, noclick_target], dim=0)
        )
        return loss

class MyInterestEvolving(nn.Module):
    """Replacement for InterestEvolving that uses MyGRU/MyDynamicGRU and avoids pack_padded_sequence.

    Supports: 'GRU', 'AIGRU', 'AGRU', 'AUGRU'
    """
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']
    def __init__(self, input_size: int, gru_type: str = 'GRU', use_neg: bool = False, init_std: float = 0.001,
                 att_hidden_size=(64, 16), att_activation='sigmoid',
                 att_weight_normalization=False, device: str = 'cpu'):
        super(MyInterestEvolving, self).__init__()
        if gru_type not in MyInterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError(f"gru_type: {gru_type} is not supported")
        self.gru_type = gru_type
        self.use_neg = use_neg
        # attention layer must be provided from user (keeps original API). If unavailable create a placeholder.
        if AttentionSequencePoolingLayer is not None:
            return_score = (gru_type != 'GRU')
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size, att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=return_score)
        else:
            self.attention = None

        if gru_type == 'GRU' or gru_type == 'AIGRU':
            self.interest_evolution = MyGRU(input_size, input_size)
        else:  # AGRU / AUGRU
            self.interest_evolution = MyDynamicGRU(input_size, input_size, gru_type=gru_type)
        self.to(device)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, keys_length: torch.LongTensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query: [B, H], keys: [B, T, H], keys_length: [B]
        bias_b, dim = query.size()
        device = query.device
        zero_outputs = torch.zeros(bias_b, dim, device=device, dtype=query.dtype)
        valid_mask = keys_length > 0
        if valid_mask.sum() == 0:
            return zero_outputs

        # select only valid rows
        q = query[valid_mask]
        keys_sel = keys[valid_mask]
        lengths_sel = keys_length[valid_mask]

        # query -> [b, 1, H]
        q_unsq = q.unsqueeze(1)

        # attention behavior
        if self.attention is None:
            # fallback: uniform attention
            att_scores = torch.ones((keys_sel.size(0), keys_sel.size(1)), device=device, dtype=query.dtype)
        else:
            att_out = self.attention(q_unsq, keys_sel, lengths_sel.unsqueeze(1))
            # att_out: if return_score True -> [b, 1, T] or if False -> [b, 1, H]
            if self.gru_type == 'GRU':
                # attention returns pooled result [b,1,H]
                # need to run GRU and then apply attention differently: we will run GRU then apply attention on outputs
                outputs = self.interest_evolution(keys_sel, lengths_sel)
                # run attention with original API to get pooled output
                pooled = self.attention(q_unsq, outputs, lengths_sel.unsqueeze(1))
                res = pooled.squeeze(1)
                zero_outputs[valid_mask] = res
                return zero_outputs
            else:
                # return_score True -> att_out is [b,1,T] -> squeeze to [b,T]
                att_scores = att_out.squeeze(1)

        if self.gru_type == 'AIGRU':
            # AIGRU: multiply keys by attention scores and run MyGRU
            keys_att = keys_sel * att_scores.unsqueeze(2)
            outputs = self.interest_evolution(keys_att, lengths_sel)
            last = MyInterestEvolving._get_last_state(outputs, lengths_sel)
        elif self.gru_type in ('AGRU', 'AUGRU'):
            # AGRU/AUGRU: dynamic GRU that consumes attention scores
            outputs = self.interest_evolution(keys_sel, att_scores, lengths_sel)
            last = MyInterestEvolving._get_last_state(outputs, lengths_sel)
        else:
            # GRU handled earlier, but keep safe fallback
            outputs = self.interest_evolution(keys_sel, lengths_sel)
            last = MyInterestEvolving._get_last_state(outputs, lengths_sel)

        zero_outputs[valid_mask] = last
        return zero_outputs

    def _get_last_state(self, states: torch.Tensor, keys_length: torch.LongTensor) -> torch.Tensor:
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()
        idx = (keys_length - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=states.device)
        return states[batch_idx, idx]
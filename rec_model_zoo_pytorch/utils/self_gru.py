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

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from deepctr_torch.layers import DNN, AttentionSequencePoolingLayer
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        self.dropout = float(dropout) if num_layers > 1 else 0.0

        self.params = nn.ModuleList()
        for layer in range(num_layers):
            layer_params = nn.ModuleList()
            for d in range(self.num_directions):
                layer_input = input_size if layer == 0 else hidden_size * self.num_directions
                p = nn.ParameterDict(
                    {
                        "W_ih": nn.Parameter(torch.Tensor(3 * hidden_size, layer_input)),
                        "W_hh": nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size)),
                        "b_ih": nn.Parameter(torch.Tensor(3 * hidden_size)),
                        "b_hh": nn.Parameter(torch.Tensor(3 * hidden_size)),
                    }
                )
                layer_params.append(p)
            self.params.append(layer_params)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for layer in self.params:
            for p in layer:
                for key in p:
                    p[key].data.uniform_(-stdv, stdv)

    def forward(self, x, lengths=None, h0=None):
        # 检查输入是否为PackedSequence
        is_packed = isinstance(x, PackedSequence)

        if is_packed:
            # 如果是PackedSequence，先解包
            x, lengths = pad_packed_sequence(x, batch_first=self.batch_first)

        if not self.batch_first:
            x = x.transpose(0, 1)

        batch, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        if lengths is None:
            lengths = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device)

        if h0 is None:
            h = torch.zeros(self.num_layers * self.num_directions, batch, self.hidden_size, dtype=dtype, device=device)
        else:
            h = h0.to(device)

        prev_out = x
        next_h_list = []

        for l_idx, layer_params in enumerate(self.params):
            layer_outs = []

            for d_idx, p in enumerate(layer_params):
                is_reverse = d_idx == 1
                h_t = h[l_idx * self.num_directions + d_idx]

                gi = torch.matmul(prev_out, p["W_ih"].T) + p["b_ih"].unsqueeze(0)
                if is_reverse:
                    gi = torch.flip(gi, [1])

                h_seq = []

                for t in range(seq_len):
                    gi_t = gi[:, t, :]
                    gh_t = torch.matmul(h_t, p["W_hh"].T) + p["b_hh"].unsqueeze(0)
                    gi_r, gi_z, gi_n = gi_t.chunk(3, dim=1)
                    gh_r, gh_z, gh_n = gh_t.chunk(3, dim=1)

                    r = torch.sigmoid(gi_r + gh_r)
                    z = torch.sigmoid(gi_z + gh_z)
                    n = torch.tanh(gi_n + r * gh_n)

                    h_t = (1 - z) * n + z * h_t

                    # mask
                    mask = (t < lengths).float().unsqueeze(1)
                    h_t = h_t * mask + h_t.detach() * (1 - mask)  # 使用detach()防止梯度传播
                    h_seq.append(h_t)

                h_seq = torch.stack(h_seq, dim=1)
                if is_reverse:
                    h_seq = torch.flip(h_seq, [1])

                layer_outs.append(h_seq)
                next_h_list.append(h_t)

            if self.num_directions == 1:
                combined = layer_outs[0]
            else:
                combined = torch.cat(layer_outs, dim=2)

            if self.dropout > 0 and l_idx < self.num_layers - 1:
                combined = nn.functional.dropout(combined, p=self.dropout, training=self.training)

            prev_out = combined

        outputs = prev_out
        h_n = torch.stack(next_h_list, dim=0)

        # 如果输入是PackedSequence，将输出重新打包
        if is_packed:
            if not self.batch_first:
                outputs = outputs.transpose(0, 1)
            outputs = pack_padded_sequence(outputs, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        elif not self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, h_n


class MyExtractor(nn.Module):
    def __init__(self, input_size, use_neg=False, init_std=0.001, device="cpu"):
        super(MyExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = CustomGRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        if self.use_neg:
            self.auxiliary_net = DNN(input_size * 2, [100, 50, 1], "sigmoid", init_std=init_std, device=device)
        for name, tensor in self.gru.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)

    def forward(self, keys, keys_length, neg_keys=None):
        """
        Parameters
        ----------
        keys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]
        neg_keys: 3D tensor, [B, T, H]

        Returns
        -------
        masked_interests: 2D tensor, [b, H]
        aux_loss: [1]
        """
        batch_size, max_length, dim = keys.size()
        zero_outputs = torch.zeros(batch_size, dim, device=keys.device)
        aux_loss = torch.zeros((1,), device=keys.device)

        # create zero mask for keys_length, to make sure 'pack_padded_sequence' safe
        mask = keys_length > 0
        masked_keys_length = keys_length[mask]

        # batch_size validation check
        if masked_keys_length.shape[0] == 0:
            return (zero_outputs,)

        masked_keys = torch.masked_select(keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)

        packed_keys = pack_padded_sequence(
            masked_keys, lengths=masked_keys_length.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(
            packed_interests, batch_first=True, padding_value=0.0, total_length=max_length
        )

        if self.use_neg and neg_keys is not None:
            masked_neg_keys = torch.masked_select(neg_keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
            aux_loss = self._cal_auxiliary_loss(
                interests[:, :-1, :], masked_keys[:, 1:, :], masked_neg_keys[:, 1:, :], masked_keys_length - 1
            )

        return interests, aux_loss

    def _cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
        # keys_length >= 1
        mask_shape = keys_length > 0
        keys_length = keys_length[mask_shape]
        if keys_length.shape[0] == 0:
            return torch.zeros((1,), device=states.device)

        _, max_seq_length, embedding_size = states.size()
        states = torch.masked_select(states, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        click_seq = torch.masked_select(click_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        noclick_seq = torch.masked_select(noclick_seq, mask_shape.view(-1, 1, 1)).view(
            -1, max_seq_length, embedding_size
        )
        batch_size = states.size()[0]

        mask = (
            torch.arange(max_seq_length, device=states.device).repeat(batch_size, 1) < keys_length.view(-1, 1)
        ).float()

        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2

        click_p = (
            self.auxiliary_net(click_input.view(batch_size * max_seq_length, embedding_size))
            .view(batch_size, max_seq_length)[mask > 0]
            .view(-1, 1)
        )
        click_target = torch.ones(click_p.size(), dtype=torch.float, device=click_p.device)

        noclick_p = (
            self.auxiliary_net(noclick_input.view(batch_size * max_seq_length, embedding_size))
            .view(batch_size, max_seq_length)[mask > 0]
            .view(-1, 1)
        )
        noclick_target = torch.zeros(noclick_p.size(), dtype=torch.float, device=noclick_p.device)

        loss = F.binary_cross_entropy(
            torch.cat([click_p, noclick_p], dim=0), torch.cat([click_target, noclick_target], dim=0)
        )

        return loss


class MyInterestEvolving(nn.Module):
    __SUPPORTED_GRU_TYPE__ = ["GRU", "AIGRU", "AGRU", "AUGRU"]

    def __init__(
        self,
        input_size,
        gru_type="GRU",
        use_neg=False,
        init_std=0.001,
        att_hidden_size=(64, 16),
        att_activation="sigmoid",
        device="cpu",
        att_weight_normalization=False,
    ):
        super(MyInterestEvolving, self).__init__()
        if gru_type not in MyInterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError("gru_type: {gru_type} is not supported")
        self.gru_type = gru_type
        self.use_neg = use_neg

        if gru_type == "GRU":
            self.attention = AttentionSequencePoolingLayer(
                embedding_dim=input_size,
                att_hidden_units=att_hidden_size,
                att_activation=att_activation,
                weight_normalization=att_weight_normalization,
                return_score=False,
            )
            self.interest_evolution = CustomGRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == "AIGRU":
            self.attention = AttentionSequencePoolingLayer(
                embedding_dim=input_size,
                att_hidden_units=att_hidden_size,
                att_activation=att_activation,
                weight_normalization=att_weight_normalization,
                return_score=True,
            )
            self.interest_evolution = CustomGRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == "AGRU" or gru_type == "AUGRU":
            self.attention = AttentionSequencePoolingLayer(
                embedding_dim=input_size,
                att_hidden_units=att_hidden_size,
                att_activation=att_activation,
                weight_normalization=att_weight_normalization,
                return_score=True,
            )
            self.interest_evolution = CustomGRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        for name, tensor in self.interest_evolution.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        # self.to(device)

    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()

        mask = torch.arange(max_seq_length, device=keys_length.device).repeat(batch_size, 1) == (
            keys_length.view(-1, 1) - 1
        )

        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        keys: (masked_interests), 3D tensor, [b, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, [B, H]
        """
        batch_size, dim = query.size()
        max_length = keys.size()[1]

        # check batch validation
        zero_outputs = torch.zeros(batch_size, dim, device=query.device)
        mask = keys_length > 0
        # [B] -> [b]
        keys_length = keys_length[mask]
        if keys_length.shape[0] == 0:
            return zero_outputs

        # [B, H] -> [b, 1, H]
        query = torch.masked_select(query, mask.view(-1, 1)).view(-1, dim).unsqueeze(1)

        if self.gru_type == "GRU":
            packed_keys = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(
                packed_interests, batch_first=True, padding_value=0.0, total_length=max_length
            )
            outputs = self.attention(query, interests, keys_length.unsqueeze(1))  # [b, 1, H]
            outputs = outputs.squeeze(1)  # [b, H]
        elif self.gru_type == "AIGRU":
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1))  # [b, 1, T]
            interests = keys * att_scores.transpose(1, 2)  # [b, T, H]
            packed_interests = pack_padded_sequence(
                interests, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False
            )
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)  # [b, H]
        elif self.gru_type == "AGRU" or self.gru_type == "AUGRU":
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)).squeeze(1)  # [b, T]
            packed_interests = pack_padded_sequence(
                keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_scores = pack_padded_sequence(
                att_scores, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=max_length)
            # pick last state
            outputs = MyInterestEvolving._get_last_state(outputs, keys_length)  # [b, H]
        else:
            raise ValueError(f"invalid gru_type, only support: GRU/AIGRU/AUGRU")
        # [b, H] -> [B, H]
        zero_outputs[mask] = outputs
        return zero_outputs

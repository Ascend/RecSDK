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
import sys
import os
from typing import Dict

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from datasets.criteo import load_data, TestCriteoHandler
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed


class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CustomMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, seq_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, seq_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        # 应用mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # 输出投影
        output = self.w_o(attn_output)
        return output, attn_weights


class MultiheadAttentionWithResidual(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.residual_fc = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.residual_fc.weight, std=0.1)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        residual = self.residual_fc(x)
        output = attn_output + residual
        return F.relu(output)


class AutoIntPlus(nn.Module):
    def __init__(self, params):
        super(AutoIntPlus, self).__init__()

        # Model configuration
        self.field_size = params.field_size
        self.vocab_size = params.vocab_size
        self.embedding_size = params.embedding_size
        self.attention_layers = params.attention_layers
        self.heads_num = params.heads_num
        self.learning_rate = params.learning_rate
        self.deep_layers = params.deep_layers

        # Parse deep layers
        self.layers = (
            list(map(int, self.deep_layers.split(","))) if self.deep_layers else []
        )

        # ------ Embeddings (matching TF initialization) ------
        self.feat_emb_deep = nn.Embedding(self.vocab_size, self.embedding_size)
        nn.init.normal_(
            self.feat_emb_deep.weight, mean=0, std=0.1
        )  # Matching TF's random_normal_initializer

        # ------ Multihead Attention Layers ------
        self.attention_layers_list = nn.ModuleList()
        for i in range(self.attention_layers):
            self.attention_layers_list.append(
                MultiheadAttentionWithResidual(self.embedding_size, self.heads_num)
            )

        # ------ Fully Connected Layer ------
        self.fc_layer = nn.Linear(self.field_size * self.embedding_size, 1)
        nn.init.normal_(
            self.fc_layer.weight, mean=0, std=0.1
        )  # Matching TF initialization

        # ------ Deep Layers ------
        self.deep_layers_list = nn.ModuleList()
        current_size = self.field_size * self.embedding_size

        # Add deep layers
        for layer_i, layer_size in enumerate(self.layers):
            layer = nn.Linear(current_size, layer_size)
            nn.init.normal_(layer.weight, mean=0, std=0.1)  # Matching TF initialization
            self.deep_layers_list.append(layer)
            current_size = layer_size

        # Final output layer for Deep
        self.deep_output = nn.Linear(current_size, 1)
        nn.init.normal_(
            self.deep_output.weight, mean=0, std=0.1
        )  # Matching TF initialization
        self.cross_loss = nn.CrossEntropyLoss()

    def embedding_layer(self, feat_ids, feat_vals):
        """Embedding layer matching TF implementation"""
        # Shape: [batch_size, field_size]
        embeddings_origin = self.feat_emb_deep(
            feat_ids
        )  # [batch_size, field_size, embedding_size]
        feat_vals = feat_vals.unsqueeze(-1)  # [batch_size, field_size, 1]
        embeddings = (
            embeddings_origin * feat_vals
        )  # [batch_size, field_size, embedding_size]
        return embeddings

    def forward(self, features: Dict[str, torch.Tensor], mode="train") -> torch.Tensor:
        """Forward pass matching TF implementation"""
        # Get feature IDs and values
        feat_ids = features["feat_ids"]  # [batch_size, field_size]
        feat_vals = features["feat_vals"]  # [batch_size, field_size]

        # ------ Embedding Layer ------
        embeddings = self.embedding_layer(
            feat_ids, feat_vals
        )  # [batch_size, field_size, embedding_size]

        # ------ Multihead Attention Layers ------
        attention_part = embeddings
        for layer in self.attention_layers_list:
            attention_part = layer(attention_part)

        # ------ FC Layer ------
        fc_inputs = attention_part.reshape(-1, self.field_size * self.embedding_size)
        y = self.fc_layer(fc_inputs).squeeze(-1)  # [batch_size]

        # ------ Deep Layers ------
        deep_inputs = embeddings.reshape(-1, self.field_size * self.embedding_size)
        for layer in self.deep_layers_list:
            deep_inputs = layer(deep_inputs)
            deep_inputs = F.relu(deep_inputs)  # ReLU activation after each layer

        y_mlp = self.deep_output(deep_inputs).squeeze(-1)  # [batch_size]

        # ------ Combine Outputs ------
        y += y_mlp  # Residual connection

        return {"ctr": torch.sigmoid(y)}

    def loss(self, pred, labels):
        return self.cross_loss(pred["ctr"], labels)


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "field_size": 39,
                "vocab_size": 2100000,
                "attention_layers": 3,
                "deep_layers": "400,400,400",
                "heads_num": 2,
                "model": "autoint_plus",
            }
        )
    )
    params = get_opts(sys.argv, params)
    set_all_seed(params)
    model = AutoIntPlus(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params, model, optimizer, load_data, TestCriteoHandler(params)
    )
    handler.run()

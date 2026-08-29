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
import getopt
from typing import Dict

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from datasets.aliccp import load_data, TestAliccpHandler, get_spec
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed
from utils.logger import logger


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


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, att_embedding_size):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att_embedding_size = att_embedding_size

        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # self.mha = CustomMultiheadAttention(embed_dim, num_heads)
      
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """Transformer block with residual connections"""
        # Multi-head attention
        attn_output, _ = self.mha(
            x, x, x, key_padding_mask=(mask == 0) if mask is not None else None
        )
        x = self.ln1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        return x


class FieldAttentionUnit(nn.Module):
    def __init__(self, embedding_size, attention_layers):
        super(FieldAttentionUnit, self).__init__()
        self.embedding_size = embedding_size

        # Attention MLP layers
        self.attention_layers = nn.ModuleList()
        input_size = 4 * embedding_size  # concat [a, b, a-b, a*b]

        for layer_size in attention_layers:
            self.attention_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size

        # Attention output layer
        self.att_out = nn.Linear(input_size, 1)

    def forward(self, target_emb, history_emb, history_len=None):
        """
        Field-wise attention unit

        Args:
            target_emb: [batch_size, 1, embedding_size]
            history_emb: [batch_size, seq_len, embedding_size]
            history_len: [batch_size, 1] (sequence lengths)
        """
        batch_size, seq_len, _ = history_emb.size()

        # Tile target embedding to match sequence length
        target_tiled = target_emb.expand(-1, seq_len, -1)  # [batch_size, seq_len, emb]

        # Concatenate features: [a, b, a-b, a*b]
        x_inputs = torch.cat(
            [
                target_tiled,
                history_emb,
                target_tiled - history_emb,
                target_tiled * history_emb,
            ],
            dim=-1,
        )  # [batch_size, seq_len, 4*emb]

        # Attention MLP
        for layer in self.attention_layers:
            x_inputs = F.leaky_relu(layer(x_inputs))

        # Attention weights
        att_wgt = self.att_out(x_inputs)  # [batch_size, seq_len, 1]
        att_wgt = att_wgt.squeeze(-1)  # [batch_size, seq_len]

        # Apply mask if provided
        if history_len is not None:
            mask = (
                torch.arange(seq_len).expand(batch_size, seq_len).to(history_len.device)
                < history_len
            )
            att_wgt = att_wgt.masked_fill(~mask, -1e9)

        # Softmax attention
        att_wgt = F.softmax(
            att_wgt / torch.sqrt(torch.tensor(self.embedding_size)), dim=-1
        )

        # Weighted sum
        wgt_emb = torch.sum(att_wgt.unsqueeze(-1) * history_emb, dim=1, keepdim=True)
        return wgt_emb


class BST(nn.Module):
    def __init__(self, params: edict, spec):
        """
        PyTorch implementation of BST (Behavior Sequence Transformer) model

        Args:
            config: Model configuration dictionary containing:
                embedding_size: Embedding dimension
                transformer_layers: Number of transformer layers
                att_embedding_size: Attention embedding size
                heads_num: Number of attention heads
                attention_layers: Comma-separated string of attention MLP layer sizes
                deep_layers: Comma-separated string of deep layer sizes
                vocab_length: Dictionary of feature vocab sizes
        """
        super(BST, self).__init__()
        self.params = params
        self.spec = spec

        # Parse layer sizes
        self.attention_layers = (
            list(map(int, params["attention_layers"]))
            if params["attention_layers"]
            else []
        )
        self.deep_layers = (
            list(map(int, params["deep_layers"])) if params["deep_layers"] else []
        )

        # ------ Embedding Layer ------
        self.emb_weights = nn.ModuleDict()
        for key, vocab_len in self.spec["vocab_length"].items():
            self.emb_weights[key] = nn.Embedding(
                num_embeddings=vocab_len + 1,
                embedding_dim=params["embedding_size"],
                padding_idx=0,
            )
            # Initialize with stddev=(2/512)**0.5 ≈ 0.0625
            nn.init.normal_(self.emb_weights[key].weight, mean=0, std=0.0625)

        # ------ Transformer Layers ------
        self.transformer_layers = nn.ModuleList()
        for _ in range(params["transformer_layers"]):
            self.transformer_layers.append(
                TransformerBlock(
                    embed_dim=params["embedding_size"],
                    num_heads=params["heads_num"],
                    att_embedding_size=params["att_embedding_size"],
                )
            )

        # ------ Field-wise Pooling Layers ------
        self.field_attention = nn.ModuleDict()
        attention_pairs = [
            ("206", "109_14"),
            ("207", "110_14"),
            ("216", "127_14"),
            ("210", "150_14"),
        ]

        for target_key, his_key in attention_pairs:
            self.field_attention[f"{target_key}_{his_key}"] = FieldAttentionUnit(
                params["embedding_size"], self.attention_layers
            )

        # ------ MLP Layers ------
        self.mlp_layers = nn.ModuleList()
        input_size = 23 * params["embedding_size"]  # 23 fields as in original
        for layer_size in self.deep_layers:
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size

        # ------ Output Layer ------
        self.output_layer = nn.Linear(input_size, 1)

        # ------ Positional Encoding ---------
        self.positional_embeddings = nn.ModuleDict()
        for key in ["109_14", "110_14", "127_14", "150_14"]:
            # 获取序列最大长度（从配置中获取或使用默认值）
            max_seq_len = params.get("max_seq_len", 20)  # 假设最大序列长度为20
            self.positional_embeddings[key] = nn.Embedding(
                num_embeddings=max_seq_len, embedding_dim=params["embedding_size"]
            )
            # 使用Xavier初始化匹配原始实现
            nn.init.xavier_uniform_(self.positional_embeddings[key].weight)

    def build_embedding_layer(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Build the embedding layer"""

        embeddings = {}
        dense_len = {}
        # One-hot fields
        for key in self.spec["one_hot_fields"]:
            embeddings[key] = self.emb_weights[key](features[key]).unsqueeze(1)

        # Multi-hot field
        for key in self.spec["special_fields"]:
            embeddings[key] = self._sparse_embedding(key, features.get(key))

        # Sequence fields
        for key in self.spec["multi_hot_fields"]:
            feature_dense = features[key]
            mask = (feature_dense >= 0).float()
            #dense_len[key] = torch.sum((feature_dense >= 0).int(), dim=1, keepdim=True)
            dense_len[key] = mask
            # Replace -1 with 0 for embedding lookup
            feature_dense = torch.where(
                feature_dense >= 0, feature_dense, torch.zeros_like(feature_dense)
            )
            emb = self.emb_weights[key](feature_dense)
            embeddings[key] = emb * mask.unsqueeze(-1)
      
        return embeddings, dense_len

    def _sparse_embedding(self, key, values):
        """Simulate sparse embedding with sum combiner"""
        if values is None:
            return torch.zeros(1, 1, self.params["embedding_size"])

        # Create mask for valid entries
        mask = (values >= 0).float()
        # Replace -1 with 0 for embedding lookup
        values = torch.where(values >= 0, values, torch.zeros_like(values))
        emb = self.emb_weights[key](values)
        # Apply mask and sum along sequence dimension
        return (emb * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)

    def positional_encoding_learn(self, inputs, key):
        """Learnable positional encoding using pre-defined parameters"""
        batch_size, seq_len, emb_dim = inputs.size()

        # 创建位置索引 [0, 1, 2, ..., seq_len-1]
        position_ind = (
            torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(inputs.device)
        )

        # 从预定义参数获取位置编码
        pos_emb = self.positional_embeddings[key](position_ind)

        # 缩放并添加到输入
        pos_emb = pos_emb * (emb_dim**0.5)
        return inputs + pos_emb

    def forward(self, features: Dict[str, torch.Tensor], mode="train") -> torch.Tensor:
        """Forward pass of BST model"""
        # 1. Build embeddings
        embeddings, dense_len = self.build_embedding_layer(features)

        # 2. Transformer layers for sequence features
        for key in self.spec['multi_hot_fields']:
            feature_dense = features.get(key)
            #mask = (feature_dense >= 0).float().unsqueeze(-1).unsqueeze(-1)
            # Positional encoding
            embeddings[key] = self.positional_encoding_learn(embeddings[key], key)
            # Transformer layers
            for transformer in self.transformer_layers:
                #print(embeddings[key].shape, dense_len.get(key).shape)
                embeddings[key] = transformer(embeddings[key], dense_len.get(key))
                #embeddings[key] = transformer(embeddings[key], dense_len.get(key).unsqueeze(1))
      
        # 3. Field-wise pooling
        for target_key, his_key in [
            ("206", "109_14"),
            ("207", "110_14"),
            ("216", "127_14"),
            ("210", "150_14"),
        ]:
            attention_key = f"{target_key}_{his_key}"
            embeddings[his_key] = self.field_attention[attention_key](
                embeddings[target_key], embeddings[his_key], dense_len.get(his_key)
            )

        # 4. Concatenate all embeddings
        concat_emb = torch.cat(
            [embeddings[k] for k in self.spec["one_hot_fields"]]
            + [embeddings[k] for k in self.spec["multi_hot_fields"]]
            + [embeddings[k] for k in self.spec["special_fields"]],
            dim=1,
        )

        # 5. MLP layers
        x_deep = concat_emb.view(concat_emb.size(0), -1)  # Flatten
        for layer in self.mlp_layers:
            x_deep = F.leaky_relu(layer(x_deep))

        # 6. Output layer
        y = self.output_layer(x_deep).squeeze(-1)
        y = torch.sigmoid(y)
        return {"ctr": y}

    def loss(self, pred, labels):
        pred_ctr = pred["ctr"]
        y = labels["y"]
        epsilon = 1e-7
        click_weight = 0.14

        ctr_loss = -(1 - click_weight) / click_weight * y * torch.log(
            pred_ctr + epsilon
        ) - (1 - y) * torch.log(1 - pred_ctr + epsilon)

        loss = torch.mean(ctr_loss)
        return loss


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "heads_num": 4,
                "transformer_layers": 1,
                "att_embedding_size": 4,
                "attention_layers": [80, 40],
                "deep_layers": [512, 256, 128, 64],
                "model": "bst",
            }
        )
    )
    params = get_opts(sys.argv, params)
    set_all_seed(params)
    # params.deep_layers = [val * 6 for val in params.deep_layers]
    # params.embedding_size = [8, 16, 32, 64, 128]
    logger.info(params)

    spec = get_spec(params)
    model = BST(params, spec).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params, model, optimizer, load_data, TestAliccpHandler(params, spec)
    )
    handler.run()

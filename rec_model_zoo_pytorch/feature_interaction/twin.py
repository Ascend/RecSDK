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

# 动态添加项目根目录到 Python 路径
try:
    root_path = os.path.abspath(__file__)
    root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
    if root_path not in sys.path:
        sys.path.append(root_path)
except NameError:
    if "." not in sys.path:
        sys.path.append(".")

# 导入框架的公共模块和库
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from datasets.criteo import load_data, TestCriteoHandler
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed


# ============== Helper Modules for TWIN ==============
# TWIN 包含两种不同的注意力机制，我们需要将它们重构为独立的 PyTorch 模块


class MultiHeadTargetAttention(nn.Module):
    """
    标准的 Target Attention 模块 (类似于 DIN)。
    用于捕捉短期序列中与目标物品相关的兴趣。
    """

    def __init__(self, input_dim, attention_dim, num_heads, dropout_rate=0.0):
        super(MultiHeadTargetAttention, self).__init__()
        assert (
            attention_dim % num_heads == 0
        ), "attention_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_o = nn.Linear(attention_dim, input_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, target_item_emb, history_seq_emb, mask=None):
        # target_item_emb: [batch, input_dim]
        # history_seq_emb: [batch, seq_len, input_dim]
        # mask: [batch, seq_len]

        batch_size, seq_len, _ = history_seq_emb.size()

        # 线性变换
        query = (
            self.W_q(target_item_emb)
            .view(batch_size, 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.W_k(history_seq_emb)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.W_v(history_seq_emb)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            mask = mask.view(batch_size, 1, 1, seq_len).expand_as(scores)
            scores = scores.masked_fill(mask.float() == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        if self.dropout:
            attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, value)

        # 合并多头
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        output = self.W_o(output.squeeze(1))  # [batch, input_dim]

        return output


class MultiHeadTopKAttention(nn.Module):
    """
    独特的 TopK 注意力机制，用于从长序列中检索最相关的兴趣。
    """

    def __init__(self, input_dim, attention_dim, topk, num_heads, dropout_rate=0.0):
        super(MultiHeadTopKAttention, self).__init__()
        assert (
            attention_dim % num_heads == 0
        ), "attention_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.topk = topk
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim**0.5

        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_o = nn.Linear(attention_dim, input_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, target_item, item_sequence, mask=None):
        batch_size, seq_len, _ = item_sequence.size()

        # 线性变换
        query = (
            self.W_q(target_item)
            .view(batch_size, 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.W_k(item_sequence)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.W_v(item_sequence)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 计算分数
        scores = (
            torch.matmul(query, key.transpose(-1, -2)) / self.scale
        )  # [b, h, 1, len]

        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand_as(scores)
            scores = scores.masked_fill(mask.float() == 0, -1e9)

        # 核心区别：Top-K 操作
        effective_topk = min(self.topk, seq_len)
        topk_scores, topk_indices = scores.topk(
            effective_topk, dim=-1
        )  # [b, h, 1, topk]
        topk_indices = topk_indices.squeeze(2)   # [b, h, topk]

        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  # [b,h,topk,head_dim]

        topk_value = torch.gather(value, 2, topk_indices)

        # # 从 value 中收集 topk 对应的向量
        # topk_value = torch.gather(
        #     value, 3, topk_indices.transpose(-1, -2).expand(-1, -1, -1, self.head_dim)
        # )

        attention = F.softmax(topk_scores, dim=-1)  # [b, h, 1, topk]

        if self.dropout:
            attention = self.dropout(attention)

        output = torch.matmul(attention, topk_value)  # [b, h, 1, head_dim]

        # 合并多头
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.num_heads * self.head_dim)
        )
        output = self.W_o(output)
        return output


# ============== Main Model: TWIN ==============
class TWIN(nn.Module):
    def __init__(self, params):
        super(TWIN, self).__init__()
        # 1. 加载参数
        self.vocab_size = params.vocab_size
        self.embedding_size = params.embedding_size
        self.max_seq_len = params.max_seq_len
        self.short_seq_len = params.short_seq_len
        self.dnn_hidden_units = [int(u) for u in params.dnn_hidden_units.split(",")]
        self.attention_dim = params.attention_dim
        self.num_heads = params.num_heads
        self.attention_dropout = params.attention_dropout
        self.net_dropout = params.net_dropout
        self.batch_norm = params.batch_norm
        self.topk = params.topk

        # 2. 定义模型层
        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=0
        )

        # 短期兴趣注意力
        self.short_attention = MultiHeadTargetAttention(
            self.embedding_size,
            self.attention_dim,
            self.num_heads,
            self.attention_dropout,
        )
        # 长期兴趣注意力
        self.long_attention = MultiHeadTopKAttention(
            self.embedding_size,
            self.attention_dim,
            self.topk,
            self.num_heads,
            self.attention_dropout,
        )

        # 最终的 DNN
        # 输入维度 = TargetEmb + ShortInterestEmb + LongInterestEmb
        input_dim = self.embedding_size * 3
        mlp_layers = []
        for hidden_size in self.dnn_hidden_units:
            mlp_layers.append(nn.Linear(input_dim, hidden_size))
            if self.batch_norm:
                mlp_layers.append(nn.BatchNorm1d(hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(self.net_dropout))
            input_dim = hidden_size
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*mlp_layers)

        # 3. 定义损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, mode="train"):
        feat_ids = features["feat_ids"]  # shape: [batch, max_seq_len]
        # print("feat_ids shape:", feat_ids.shape)
        # print("feat_ids max:", feat_ids.max().item())
        # print("vocab_size:", self.vocab_size)
        #assert feat_ids.max() < self.vocab_size, f"ID {feat_ids.max()} >= vocab_size {self.vocab_size}"
        #assert feat_ids.min() >= 0, f"Negative ID found: {feat_ids.min()}"
        # 创建 padding mask
        mask = feat_ids != 0  # shape: [batch, max_seq_len]

        # 提取目标物品和历史序列 (与 TransAct 假设一致)
        seq_lengths = mask.sum(dim=1)
        safe_lengths = torch.clamp(seq_lengths - 1, min=0)
        target_item_indices = safe_lengths.unsqueeze(1)
        target_item_id = torch.gather(feat_ids, 1, target_item_indices)

        # 获取所有 embedding
        item_feat_emb = self.embedding_layer(feat_ids)  # [batch, max_seq_len, emb_size]
        target_emb = self.embedding_layer(target_item_id).squeeze(
            1
        )  # [batch, emb_size]

        # 1. 短期兴趣提取
        # 从完整序列中截取短期部分
        short_seq_emb = item_feat_emb[:, -self.short_seq_len :, :]
        short_mask = mask[:, -self.short_seq_len :]
        short_interest_emb = self.short_attention(target_emb, short_seq_emb, short_mask)

        # 2. 长期兴趣提取
        # 使用完整序列作为历史
        long_interest_emb = self.long_attention(target_emb, item_feat_emb, mask)

        # 3. 拼接所有特征
        feature_emb = torch.cat(
            [target_emb, short_interest_emb, long_interest_emb], dim=-1
        )
        #if torch.isnan(feature_emb).any() or torch.isinf(feature_emb).any():
        #    print("feature_emb contains NaN or Inf!")
        #    print(feature_emb)
        feature_emb = feature_emb.contiguous()
        #print("has nan:", torch.isnan(feature_emb).any())
        # 4. 通过 DNN 预测
        logits = self.dnn(feature_emb)

        return {"ctr": torch.sigmoid(logits.squeeze(-1))}

    def loss(self, pred, labels):
        logits = torch.log(pred["ctr"] / (1 - pred["ctr"]))
        return self.loss_fn(logits, labels.float())


# ============== 主执行函数 ==============
if __name__ == "__main__":
    params = get_params()

    # 更新参数以匹配 twin_config/model_config.yaml 中的 `TWIN_default`
    params.update(
        edict(
            {
                "model": "twin",
                "vocab_size": 2100000,
                # --- 从 TWIN_default 配置中同步 ---
                "learning_rate": 1.0e-3,
                "batch_size": 8192,
                "embedding_dim": 4,
                "dnn_hidden_units": "64,32",
                "attention_dim": 64,
                "num_heads": 2,
                "attention_dropout": 0.0,
                "topk": 50,
                "short_seq_len": 50,
                "net_dropout": 0.0,
                "batch_norm": False,
                "epochs": 100,
                "max_len": 50,  # 对应配置文件中的 max_len
            }
        )
    )

    params = get_opts(sys.argv, params)
    set_all_seed(params)

    # 序列模型中，field_size 就是 max_seq_len
    if "max_len" in params:
        params.max_seq_len = params.max_len

    # 确保 short_seq_len 不超过 max_seq_len
    if params.short_seq_len > params.max_seq_len:
        print(
            f"Warning: short_seq_len({params.short_seq_len}) > max_seq_len({params.max_seq_len}). Clipping short_seq_len."
        )
        params.short_seq_len = params.max_seq_len

    model = TWIN(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params,
        model,
        optimizer,
        load_data,
        TestCriteoHandler(params),
    )
    handler.run()

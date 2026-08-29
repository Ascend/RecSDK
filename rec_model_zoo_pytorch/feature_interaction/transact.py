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
import math  # For positional encoding

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


# ============== Helper Modules for TransAct ==============
# TransAct 的核心是 Transformer 模块，我们先定义它。


class PositionalEncoding(nn.Module):
    """
    为序列中的 token 添加位置信息。
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    标准的 Transformer 编码器层。
    """

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = CustomMultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Multi-head Attention
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_mask)
        # Add & Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feed Forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        # Add & Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ============== Main Model: TransAct ==============
class TransAct(nn.Module):
    def __init__(self, params):
        super(TransAct, self).__init__()

        # 1. 从 params 加载配置
        self.vocab_size = params.vocab_size
        self.embedding_size = params.embedding_size
        self.field_size = params.field_size  # 在这里代表 max_seq_len
        self.num_heads = params.num_heads
        self.num_layers = params.num_layers
        self.dropout_rate = params.dropout_rate
        self.hidden_units = list(map(int, params.hidden_units.split(",")))
        self.use_residual = params.use_residual
        self.use_scale = params.use_scale

        # 2. 定义模型层
        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=0
        )
        self.pos_encoding = PositionalEncoding(
            self.embedding_size, self.dropout_rate, max_len=self.field_size + 1
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.embedding_size,
                    num_heads=self.num_heads,
                    dim_feedforward=self.embedding_size * 4,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 3. 定义最终的 MLP 分类器
        # Transformer 输出 + Target Item Embedding
        mlp_input_dim = self.embedding_size * self.field_size + self.embedding_size
        all_layers = []
        for i in range(len(self.hidden_units)):
            all_layers.append(nn.Linear(mlp_input_dim, self.hidden_units[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(self.dropout_rate))
            mlp_input_dim = self.hidden_units[i]
        all_layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*all_layers)

        # 4. 定义损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 更稳定

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, mode="train"):
        """
        前向传播 - 适配序列模型
        """
        # !! 关键假设 !!
        # 您的数据加载器目前为每个样本提供 [batch, field_size] 的 feat_ids
        # 对于序列模型 TransAct，我们做如下假设：
        # 1. `feat_ids` 代表用户的行为序列，其中每个 ID 是一个物品。
        # 2. `field_size` 代表序列的最大长度 (max_seq_len)。
        # 3. 我们将序列的最后一个非填充(non-padding)物品作为 "Target Item"，
        #    它之前的序列作为 "User History"。这是序列推荐任务的常见设定。

        feat_ids = features["feat_ids"]  # shape: [batch, max_seq_len]

        # 创建 attention mask，忽略 padding (ID=0) 的部分
        # shape: [batch, max_seq_len]
        attention_mask = feat_ids == 0  # True for padding positions

        # 提取 target item 和 history
        # 计算每个序列的真实长度
        seq_lengths = (feat_ids != 0).sum(dim=1)

        # 创建索引以提取最后一个有效项目（目标项目）
        # handle case where seq_lengths can be 0 for empty sequences
        safe_lengths = torch.clamp(seq_lengths - 1, min=0)
        target_item_indices = safe_lengths.unsqueeze(1)

        # 提取目标物品ID
        target_item_id = torch.gather(
            feat_ids, 1, target_item_indices
        )  # shape: [batch, 1]

        # 获取 embedding
        seq_emb = self.embedding_layer(
            feat_ids
        )  # shape: [batch, max_seq_len, emb_size]
        target_item_emb = self.embedding_layer(target_item_id).squeeze(
            1
        )  # shape: [batch, emb_size]

        # 如果使用 scale，则乘以 sqrt(d)
        if self.use_scale:
            seq_emb *= self.embedding_size**0.5

        # 添加位置编码
        seq_emb = self.pos_encoding(seq_emb)

        # 通过 Transformer 层
        for layer in self.transformer_layers:
            seq_emb = layer(seq_emb, src_mask=attention_mask)

        # 将 Transformer 的输出展平
        transformer_output = seq_emb.flatten(start_dim=1)

        # 拼接 Transformer 输出和目标物品 Embedding
        mlp_input = torch.cat([transformer_output, target_item_emb], dim=1)

        # 通过 MLP
        logits = self.mlp(mlp_input)

        # 在 `loss` 函数中，BCEWithLogitsLoss 会自己做 sigmoid
        # 但为了遵循输出 {'ctr': ...}，这里我们返回 sigmoid 后的概率
        return {"ctr": torch.sigmoid(logits.squeeze(-1))}

    def loss(self, pred, labels):
        # 由于我们使用了 BCEWithLogitsLoss，理论上应该传入原始 logits
        # 但 handler 依赖 `pred['ctr']`，这里我们重新计算 logits
        # 这是一个小小的妥协，以保持框架兼容性
        # 对 pred['ctr'] 取逆 sigmoid 得到 logits
        logits = torch.log(pred["ctr"] / (1 - pred["ctr"]))
        return self.loss_fn(logits, labels.float())


# ============== 主执行函数 ==============
if __name__ == "__main__":
    params = get_params()

    # 更新参数以匹配 transact_config/model_config.yaml 中的 `TransAct_default`
    params.update(
        edict(
            {
                "model": "transact",
                "vocab_size": 2100000,
                "max_seq_len": 39,
                # --- 从 TransAct_default 配置中同步 ---
                "learning_rate": 1.0e-3,
                "batch_size": 4096,
                "embedding_dim": 64,
                "hidden_units": "512,256,128",
                "num_heads": 8,
                "num_layers": 2,
                "dropout_rate": 0.1,
                "use_residual": True,
                "use_scale": True,
                "epochs": 100,
            }
        )
    )

    params = get_opts(sys.argv, params)
    set_all_seed(params)

    # 对于序列模型，field_size 通常等于 max_seq_len
    params.field_size = params.max_seq_len

    model = TransAct(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params,
        model,
        optimizer,
        load_data,
        TestCriteoHandler(params),
    )
    handler.run()

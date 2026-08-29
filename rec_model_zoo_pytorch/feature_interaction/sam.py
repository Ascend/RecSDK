# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022 FuxiCTR Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

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


class SAM2A(nn.Module):
    def __init__(self, num_fields, embedding_dim, dropout=0):
        super(SAM2A, self).__init__()
        self.W = nn.Parameter(
            torch.ones(num_fields, num_fields, embedding_dim)
        )  # f x f x d
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2))  # b x f x f
        out = S.unsqueeze(-1) * self.W  # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM2E(nn.Module):
    def __init__(self, embedding_dim, dropout=0):
        super(SAM2E, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, F.transpose(1, 2))  # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F)  # b x f x f x d
        out = S.unsqueeze(-1) * U  # b x f x f x d
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM3A(nn.Module):
    def __init__(self, num_fields, embedding_dim, use_residual=True, dropout=0):
        super(SAM3A, self).__init__()
        self.W = nn.Parameter(
            torch.ones(num_fields, num_fields, embedding_dim)
        )  # f x f x d
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2))  # b x f x f
        out = (S.unsqueeze(-1) * self.W).sum(dim=2)  # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out


class SAM3E(nn.Module):
    def __init__(self, embedding_dim, use_residual=True, dropout=0):
        super(SAM3E, self).__init__()
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.use_residual = use_residual
        if use_residual:
            self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, F):
        S = torch.bmm(F, self.K(F).transpose(1, 2))  # b x f x f
        U = torch.einsum("bnd,bmd->bnmd", F, F)  # b x f x f x d
        out = (S.unsqueeze(-1) * U).sum(dim=2)  # b x f x d
        if self.use_residual:
            out += self.Q(F)
        if self.dropout:
            out = self.dropout(out)
        return out


class SAMBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        num_fields,
        embedding_dim,
        use_residual=False,
        interaction_type="SAM2E",
        aggregation="concat",
        dropout=0,
    ):
        super(SAMBlock, self).__init__()
        self.aggregation = aggregation
        if self.aggregation == "weighted_pooling":
            self.weight = nn.Parameter(torch.ones(num_fields, 1))

        if interaction_type == "SAM2A":
            self.layers = nn.ModuleList([SAM2A(num_fields, embedding_dim, dropout)])
        elif interaction_type == "SAM2E":
            self.layers = nn.ModuleList([SAM2E(embedding_dim, dropout)])
        elif interaction_type == "SAM3A":
            self.layers = nn.ModuleList(
                [
                    SAM3A(num_fields, embedding_dim, use_residual, dropout)
                    for _ in range(num_layers)
                ]
            )
        elif interaction_type == "SAM3E":
            self.layers = nn.ModuleList(
                [SAM3E(embedding_dim, use_residual, dropout) for _ in range(num_layers)]
            )
        else:
            raise ValueError(f"interaction_type={interaction_type} not supported.")

    def forward(self, F):
        for layer in self.layers:
            F = layer(F)

        if self.aggregation == "concat":
            out = F.flatten(start_dim=1)
        elif self.aggregation == "weighted_pooling":
            out = (F * self.weight).sum(dim=1)
        elif self.aggregation == "mean_pooling":
            out = F.mean(dim=1)
        elif self.aggregation == "sum_pooling":
            out = F.sum(dim=1)
        return out


# ============== Main Model: SAM ==============
class SAM(nn.Module):
    def __init__(self, params):
        super(SAM, self).__init__()

        self.vocab_size = params.vocab_size
        self.embedding_size = params.embedding_size
        self.field_size = params.field_size
        self.interaction_type = params.interaction_type
        self.aggregation = params.aggregation
        self.num_interaction_layers = params.num_interaction_layers
        self.use_residual = params.use_residual
        self.net_dropout = params.net_dropout if "net_dropout" in params else 0.0

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_size)

        self.sam_block = SAMBlock(
            num_layers=self.num_interaction_layers,
            num_fields=self.field_size,
            embedding_dim=self.embedding_size,
            use_residual=self.use_residual,
            interaction_type=self.interaction_type,
            aggregation=self.aggregation,
            dropout=self.net_dropout,
        )

        if self.aggregation == "concat":
            if self.interaction_type in ["SAM2A", "SAM2E"]:
                fc_input_dim = self.embedding_size * (self.field_size**2)
            else:
                fc_input_dim = self.field_size * self.embedding_size
        else:
            fc_input_dim = self.embedding_size

        self.fc = nn.Linear(fc_input_dim, 1)

        self.loss_fn = nn.BCELoss()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, features, mode="train"):
        feat_ids = features["feat_ids"]
        feat_vals = features["feat_vals"].unsqueeze(-1)

        feature_emb = self.embedding_layer(feat_ids) * feat_vals

        interact_out = self.sam_block(feature_emb)

        y = self.fc(interact_out)

        return {"ctr": torch.sigmoid(y.squeeze(-1))}

    def loss(self, pred, labels):
        return self.loss_fn(pred["ctr"], labels.float())


if __name__ == "__main__":
    params = get_params()

    # ==================== 配置更新: START ====================
    # 更新参数以完全匹配 model_config.yaml 中的 `SAM_default`
    params.update(
        edict(
            {
                "model": "sam",
                "vocab_size": 2100000,  # 假设的词典大小，与您其他模型一致
                # --- 从 SAM_default 配置中同步 ---
                "learning_rate": 1.0e-3,
                "batch_size": 4096,
                "embedding_dim": 40,
                "interaction_type": "SAM2E",
                "aggregation": "concat",
                "num_interaction_layers": 3,
                "use_residual": False,
                "net_dropout": 0.0,
                "epochs": 100,
            }
        )
    )
    # ==================== 配置更新: END ======================

    params = get_opts(sys.argv, params)
    set_all_seed(params)

    # 验证参数兼容性 (原代码中的断言)
    if params.interaction_type in ["SAM2A", "SAM2E"]:
        assert (
            params.aggregation == "concat"
        ), "Only aggregation=concat is supported for SAM2A/SAM2E."

    model = SAM(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params,
        model,
        optimizer,
        load_data,
        TestCriteoHandler(params),
    )
    handler.run()

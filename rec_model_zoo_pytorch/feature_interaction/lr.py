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

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from datasets.criteo import load_data, TestCriteoHandler
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed


class LR(nn.Module):
    def __init__(self, params):
        super(LR, self).__init__()

        # 1. 从 params 对象加载模型配置
        self.vocab_size = params.vocab_size

        # 2. 定义模型层
        # 对于稀疏特征的LR，我们可以用一个 embedding_dim=1 的 Embedding 层来高效地实现。
        # 每个特征的权重就是一个维度为1的向量。
        self.embedding = nn.Embedding(self.vocab_size, 1)

        self.bias = nn.Parameter(torch.zeros(1))
        self.loss_fn = nn.BCELoss()
        self._init_weights()

    def _init_weights(self):
        # FuxiCTR 原始实现中对 LR 的权重有特定的初始化，这里我们使用简单的 xavier_uniform
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, features, mode="train"):
        # 1. 解析输入数据
        feat_ids = features["feat_ids"]  # shape: [batch_size, field_size]
        feat_vals = features["feat_vals"]  # shape: [batch_size, field_size]

        weights = self.embedding(feat_ids)

        weighted_values = weights * feat_vals.unsqueeze(-1)

        logits = torch.sum(weighted_values, dim=1) + self.bias

        return {"ctr": torch.sigmoid(logits.squeeze(-1))}

    def loss(self, pred, labels):
        return self.loss_fn(pred["ctr"], labels.float())


if __name__ == "__main__":
    # 1. 获取基础参数
    params = get_params()

    # 2. 更新模型专属参数
    # 它不需要 'embedding_size' 或 'deep_layers_dnn'。
    params.update(
        edict(
            {
                "vocab_size": 2100000,
                "model": "lr",  # <-- 指定模型名称
            }
        )
    )

    params = get_opts(sys.argv, params)
    set_all_seed(params)
    model = LR(params).to(params.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    handler = ModelHandler(
        params,
        model,
        optimizer,
        load_data,
        TestCriteoHandler(params),
    )
    handler.run()

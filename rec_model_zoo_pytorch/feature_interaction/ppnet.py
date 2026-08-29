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
# 假设此脚本位于项目的 `models` 或类似子目录中
try:
    # __file__ 在某些交互式环境（如Jupyter）中可能未定义
    root_path = os.path.abspath(__file__)
    # 将路径回退两级以到达项目根目录 (e.g., from /path/to/project/models/ppnet.py to /path/to/project)
    root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
    if root_path not in sys.path:
        sys.path.append(root_path)
except NameError:
    # 如果 __file__ 未定义，则假定当前工作目录是项目根目录
    if "." not in sys.path:
        sys.path.append(".")

# 导入框架的公共模块和库
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from datasets.criteo import load_data, TestCriteoHandler
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed


# ============== Helper Modules from Original Code ==============
# 为了让主模型代码更清晰，我们将原始代码中的辅助模块放在前面。


class GateNU(nn.Module):
    """
    门控网络单元 (Gating Network Unit)。
    这个模块接收一个输入，通过一个小型网络（通常是带有一个隐藏层的MLP），
    然后输出一个与目标张量形状相同的门控信号（gating signal）。
    输出结果会乘以 2，这是原论文中的一个细节，旨在给予门控信号更大的动态范围。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=None,
        hidden_activation="ReLU",
        dropout_rate=0.0,
    ):
        super(GateNU, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim  # 如果未指定，隐藏层维度等于输出维度

        # 定义激活函数
        if hidden_activation == "ReLU":
            activation_func = nn.ReLU()
        elif hidden_activation == "Sigmoid":
            activation_func = nn.Sigmoid()
        # 可以根据需要添加更多激活函数
        else:
            activation_func = nn.ReLU()

        layers = [nn.Linear(input_dim, hidden_dim)]
        layers.append(activation_func)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # 最后的Sigmoid确保门控信号在 (0, 1) 之间

        self.gate = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.gate(inputs) * 2


# ============== Main Model: PPNet ==============


class PPNet(nn.Module):
    def __init__(self, params):
        """
        构造函数：完全遵循你的框架接口。
        """
        super(PPNet, self).__init__()

        # 1. 从 params 对象加载模型配置
        self.vocab_size = params.vocab_size
        self.embedding_size = params.embedding_size
        self.field_size = params.field_size
        self.gate_emb_dim = params.gate_emb_dim  # 门控网络专用的 embedding 维度
        self.gate_hidden_dim = params.gate_hidden_dim  # 门控网络隐藏层维度
        self.deep_layers_str = params.deep_layers_dnn  # 复用 'deep_layers_dnn' 参数
        self.use_batch_norm = params.use_batch_norm
        self.net_dropout = (
            params.net_dropout if "net_dropout" in params else 0.0
        )  # 兼容性处理

        # 2. 解析DNN层结构
        self.deep_layers = list(map(int, self.deep_layers_str.split(",")))

        # 3. 定义 Embedding 层
        self.feat_emb = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gate_emb = nn.Embedding(self.vocab_size, self.gate_emb_dim)

        # 4. 定义 PPNet 的核心结构：带门控的 MLP
        gate_input_dim = self.field_size * (self.embedding_size + self.gate_emb_dim)

        self.mlp_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()

        current_dim = self.field_size * self.embedding_size
        for layer_size in self.deep_layers:
            mlp_layer_modules = [nn.Linear(current_dim, layer_size)]
            if self.use_batch_norm:
                mlp_layer_modules.append(nn.BatchNorm1d(layer_size))
            mlp_layer_modules.append(nn.ReLU())
            if self.net_dropout > 0:
                mlp_layer_modules.append(nn.Dropout(self.net_dropout))
            self.mlp_layers.append(nn.Sequential(*mlp_layer_modules))

            self.gate_layers.append(
                GateNU(gate_input_dim, layer_size, self.gate_hidden_dim)
            )
            current_dim = layer_size

        self.output_layer = nn.Linear(current_dim, 1)

        # 5. 定义损失函数 (建议使用 BCELoss for CTR task)
        self.loss_fn = nn.BCELoss()

        # 6. 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
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

        main_embeddings = self.feat_emb(feat_ids) * feat_vals
        gate_embeddings = self.gate_emb(feat_ids) * feat_vals

        h = main_embeddings.reshape(-1, self.field_size * self.embedding_size)

        gate_input_detached = main_embeddings.detach()
        gate_input = torch.cat([gate_input_detached, gate_embeddings], dim=-1)
        gate_input = gate_input.reshape(
            -1, self.field_size * (self.embedding_size + self.gate_emb_dim)
        )

        for i in range(len(self.deep_layers)):
            h = self.mlp_layers[i](h)
            g = self.gate_layers[i](gate_input)
            h = h * g

        y = self.output_layer(h)

        return {"ctr": torch.sigmoid(y.squeeze(-1))}

    def loss(self, pred, labels):
        return self.loss_fn(pred["ctr"], labels.float())


# ============== 主执行函数 (完全遵从您的示例) ==============
if __name__ == "__main__":
    # 1. 获取基础参数
    params = get_params()

    # 2. 更新模型专属参数
    params.update(
        edict(
            {
                # --- 通用参数 ---
                "vocab_size": 2100000,
                "embedding_size": 10,
                "deep_layers_dnn": "256,128,64",  # PPNet的MLP层结构
                "use_batch_norm": True,
                "net_dropout": 0.1,
                "model": "ppnet",  # <-- 指定模型名称
                # --- PPNet 专属参数 ---
                "gate_emb_dim": 10,  # 门控网络 embedding 维度
                "gate_hidden_dim": 64,  # 门控网络隐藏层维度
            }
        )
    )

    # 3. 从命令行覆盖参数
    params = get_opts(sys.argv, params)
    set_all_seed(params)

    # 5. 初始化模型
    model = PPNet(params).to(params.device)

    # 6. 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    # 7. 初始化训练器并运行
    handler = ModelHandler(
        params,
        model,
        optimizer,
        load_data,
        TestCriteoHandler(params),
    )
    handler.run()

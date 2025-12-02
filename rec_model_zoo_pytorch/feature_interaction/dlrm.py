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

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
from easydict import EasyDict as edict

from datasets.criteo import load_data, TestCriteoHandler
from utils.handler import ModelHandler, get_params, get_opts

class DLRM(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.feature_embedding = nn.Embedding.from_pretrained(
            torch.normal(
                mean = 0, std= 0.1, size = (params.vocab_size, params.embedding_size)
            )
        ).to(params.device)
        num_layers = [params.field_size] + params.deep_layers + [params.embedding_size]
        self.deep_layers = []
        for i in range(1, len(num_layers)):
            self.deep_layers.append(nn.Linear(num_layers[i - 1], num_layers[i]))
            self.deep_layers.append(nn.ReLU())
        self.deep_layers = nn.Sequential(*self.deep_layers)

        group_size = 20
        num_groups = (params.field_size + group_size - 1) // group_size
        num_layers = [
            (params.embedding_size + num_groups * params.embedding_size) ** 2
            + params.embedding_size
        ]

        self.predict_layers = []
        for i in range(1, len(num_layers)):
            self.predict_layers.append(nn.Linear(num_layers[i - 1], num_layers[i]))
            self.predict_layers.append(nn.ReLU())
        self.predict_layers.append(nn.Linear(num_layers[-1], 1))
        self.predict_layers = nn.Sequential(*self.predict_layers)
        self.cross_loss = nn.CrossEntropyLoss()

    def embedding_layer(self, feat_ids):
        group_size = 20
        field_size = feat_ids.shape[1]
        num_groups = (field_size + group_size - 1) // group_size
        group_sums = []
        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, field_size)
            emb = self.feature_embedding(feat_ids[:, start:end])
            group_sum = torch.sum(emb, dim=1)
            group_sums.append(group_sum)
        out = torch.cat(group_sums, dim=1)
        return out

    def dot_interaction(self, inputs):
        num_features = inputs.shape[1]
        batch_size = inputs.shape[0]
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        xactions = torch.matmul(
            inputs, inputs.transpose(1, 2)
        )
        ones = torch.ones_like(xactions, dtype = torch.float32)
        upper_tri_mask = torch.triu(ones, diagonal=0)
        activations = torch.where(
            condition = upper_tri_mask.to(bool),
            input = torch.zeros_like(xactions),
            other = xactions
        )
        activations = torch.reshape(
            activations, (batch_size, num_features * num_features)
        )
        return activations

    def forward(self, features, mode="train"):
        feat_ids = features["feat_ids"]
        feat_vals = features["feat_vals"]

        sparse_embedding = self.embedding_layer(feat_ids)
        dense_embedding = self.deep_layers(feat_vals)
        interaction_output = self.dot_interaction(
            torch.cat([dense_embedding, sparse_embedding], dim=1)
        )
        pred = self.predict_layers(
            torch.concat([dense_embedding, interaction_output], dim=-1)
        )
        return {"ctr": torch.sigmoid(pred).squeeze(dim=1)}

    def loss(self, pred, labels):
        return self.cross_loss(pred["ctr"], labels)

if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "num_heads": 4,
                "vocab_size": 5000,
                "field_size": 1000,
                "deep_layers": [
                    1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256, 256, 256
                ],
                "predict_layers": [
                    2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256, 256, 256
                ],
                "model": "dlrm"
            }
        )
    )
    params = get_opts(sys.argv, params)
    params.deep_layers = [val * 6 for val in params.deep_layers]
    params.predict_layers = [val * 4 for val in params.predict_layers]

    model = DLRM(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params, model, optimizer, load_data, TestCriteoHandler(params)
        )
    handler.run()
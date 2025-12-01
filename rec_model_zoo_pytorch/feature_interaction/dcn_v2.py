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

from datasets.criteo import load_data
from utils.handler import ModelHandler, get_params, get_opts


class Cross(nn.Module):
    def __init__(self, embedding_size, projection_dim, diag_scale=0.0) -> None:
        super().__init__()
        self.diag_scale = diag_scale
        self.project_layer = nn.Sequential(
            nn.Linear(embedding_size * 2, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, embedding_size * 2),
        )

    def forward(self, x0, x):
        out = self.project_layer(x) + self.diag_scale * x
        return x0 * out + x


class DCNV2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.feature_embedding = nn.Embedding.from_pretrained(
            torch.normal(mean=0, std=0.1, size=(params.vocab_size, params.embedding_size))
        )
        self.deep_layers = []

        num_layers = [params.field_size] + params.deep_layers + [params.embedding_size]
        for i in range(1, len(num_layers)):
            self.deep_layers.append(nn.Linear(num_layers[i - 1], num_layers[i]))
            self.deep_layers.append(nn.ReLU())
        self.deep_layers = nn.Sequential(*self.deep_layers)

        num_layers = [params.embedding_size * 3] + params.predict_layers
        self.predict_layers = nn.Sequential(
            *[nn.Linear(num_layers[idx - 1], num_layers[idx]) for idx in range(1, len(num_layers))]
        )

        self.cross_layers_list = nn.ModuleList()
        for i in range(params.num_cross_layers):
            self.cross_layers_list.append(Cross(params.embedding_size, params.cross_layer_projection_dim))

        self.ctr_loss = nn.CrossEntropyLoss()

    def embedding_layers(self, feat_ids):
        emb = self.feature_embedding(feat_ids)
        return torch.sum(emb, dim=1)

    def cross_layer(self, x0):
        x = x0
        for layer in self.cross_layers_list:
            x = layer(x0=x0, x=x)
        return x

    def forward(self, features, mode="train"):
        feat_ids = torch.reshape(features["feat_ids"], shape=(-1, self.params.field_size))
        feat_vals = torch.reshape(features["feat_vals"], shape=(-1, self.params.field_size))

        sparse_embeddings = self.embedding_layers(feat_ids)
        dense_embedding = self.deep_layers(feat_vals)
        interaction_output = self.cross_layer(torch.concat([dense_embedding, sparse_embeddings], dim=-1))
        pred = self.predict_layers(torch.concat([dense_embedding, interaction_output], dim=-1))
        return {"ctr": torch.sigmoid(pred).squeeze(dim=1)}

    def loss(self, pred, labels):
        return self.ctr_loss(pred["ctr"], labels)


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "attention_dim": 16,
                "field_size": 39,
                "vocab_size": 2100000,
                "deep_layers": [512, 216],
                "predict_layers": [1024, 1024, 512, 256, 1],
                "num_cross_layers": 3,
                "cross_layer_projection_dim": 512,
                "model": "dcn_v2",
            }
        )
    )
    params = get_opts(sys.argv, params)
    train_loader, test_loader, val_loader = load_data(params)
    model = DCNV2(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(params, model, optimizer, train_loader, test_loader, val_loader)

    handler.run()

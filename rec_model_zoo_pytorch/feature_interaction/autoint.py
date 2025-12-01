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
import torch.nn.functional as F
from easydict import EasyDict as edict

from datasets.criteo import load_data
from utils.handler import ModelHandler, get_params, get_opts


class AutoInt(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.feature_embedding = nn.Embedding.from_pretrained(
            torch.normal(mean=0, std=0.1, size=(params.vocab_size, params.embedding_size))
        )
        self.deep_layers = []

        self.attn_layers = nn.ModuleList()
        for i in range(params.attention_layers):
            self.attn_layers.append(nn.MultiheadAttention(embed_dim=params.embedding_size, num_heads=params.num_heads))

        self.fc_layer = nn.Linear(params.field_size * params.embedding_size, 1)

        self.cross_loss = nn.CrossEntropyLoss()

    def embedding_layer(self, feat_ids, feat_vals):
        emb = self.feature_embedding(feat_ids)
        return torch.multiply(emb, feat_vals.reshape((-1, self.params.field_size, 1)))

    def forward(self, features, mode="train"):
        feat_ids = torch.reshape(features["feat_ids"], shape=(-1, params.field_size))
        feat_vals = torch.reshape(features["feat_vals"], shape=(-1, params.field_size))

        embeddings = self.embedding_layer(feat_ids, feat_vals)
        for i in range(params.attention_layers):
            embeddings, _ = self.attn_layers[i](embeddings, embeddings, embeddings)
        y = self.fc_layer(embeddings.reshape(-1, self.params.field_size * self.params.embedding_size))
        return {"ctr": torch.sigmoid(y).squeeze(dim=1)}

    def loss(self, pred, labels):
        return self.cross_loss(pred["ctr"], labels)


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "attention_dim": 16,
                "field_size": 39,
                "vocab_size": 2100000,
                "attention_layers": 3,
                "num_heads": 2,
                "model": "autoint",
            }
        )
    )
    params = get_opts(sys.argv, params)
    train_loader, test_loader, val_loader = load_data(params)
    model = AutoInt(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(params, model, optimizer, train_loader, test_loader, val_loader)

    handler.run()

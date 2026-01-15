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

from datasets.criteo_multihot import *
from utils.common import get_loop_element
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed


class Cross(nn.Module):
    def __init__(self, embedding_size, projection_dim, diag_scale=0.0) -> None:
        super().__init__()
        self.diag_scale = diag_scale
        self.project_layer = nn.Sequential(
            nn.Linear(embedding_size , projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, embedding_size),
        )

    def forward(self, x0, x):
        out = self.project_layer(x) + self.diag_scale * x
        return x0 * out + x


class DCNV2MultiHot(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.feature_embedding = []
        self.check_precision = params.get("check_precision", False)
        
        idx = 0
        total_dims = 0
        for i in range(params.multi_fields_count):
            num_embeddings = get_loop_element(NUM_EMBEDDINGS_PER_FEATURE, idx)
            multi_hot = get_loop_element(MULTI_HOT_SIZES, idx)
            dim = get_loop_element(params.embedding_size, idx)
            total_dims += dim
            idx += 1
            generator = torch.Generator().manual_seed(i)
            feat_table = nn.Embedding.from_pretrained(
                torch.normal(mean=0, std=0.1, size=(num_embeddings, dim),generator=generator)
            ).to(params.device)
            if self.check_precision:
                torch.nn.init.uniform_(feat_table.weight, a=-1.0,b=1.0)
            self.feature_embedding.append(feat_table)
        
        self.deep_layers = []

        num_layers = [13] + params.deep_layers
        for i in range(1, len(num_layers)):
            self.deep_layers.append(nn.Linear(num_layers[i - 1], num_layers[i]))
            self.deep_layers.append(nn.ReLU())
        self.deep_layers = nn.Sequential(*self.deep_layers)

        num_layers = [total_dims + params.deep_layers[-1] * 2] + params.predict_layers
        self.predict_layers = nn.Sequential(
            *[
                nn.Linear(num_layers[idx - 1], num_layers[idx]) 
                for idx in range(1, len(num_layers))
            ]
        )

        self.cross_layers_list = nn.ModuleList()
        for i in range(params.num_cross_layers):
            self.cross_layers_list.append(
                Cross(
                    total_dims + params.deep_layers[-1], 
                    params.cross_layer_projection_dim
                )
            )

        self.ctr_loss = nn.CrossEntropyLoss()
        self.sigmoid = torch.nn.Sigmoid()

    def embedding_layers(self, feat_ids):
        start = 0
        features = []
        for i in range(self.params.multi_fields_count):
            length = get_loop_element(MULTI_HOT_SIZES, i)
            emb = self.feature_embedding[i](feat_ids[: , start : start + length])
            start += length
            features.append(torch.sum(emb, dim=1))

        return torch.concat(features, dim=1)

    def cross_layer(self, x0):
        x = x0
        for layer in self.cross_layers_list:
            x = layer(x0=x0, x=x)
        return x

    def forward(self, features, mode="train"):
        feat_ids = features["feat_ids"]
        feat_vals = features["feat_vals"]

        sparse_embeddings = self.embedding_layers(feat_ids)
        dense_embedding = self.deep_layers(feat_vals)
        interaction_output = self.cross_layer(torch.concat([dense_embedding, sparse_embeddings], dim=-1))
        pred = self.predict_layers(torch.concat([dense_embedding, interaction_output], dim=-1))
        pred = self.sigmoid(pred).squeeze(dim=1)
        return {"ctr": pred}

    def loss(self, pred, labels):
        return self.ctr_loss(pred["ctr"], labels)


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "num_heads": 4,
                "deep_layers": [
                    2048,2048,2048,
                    1024,1024,1024,
                    512,512,512,
                    256,256,256,
                    128,128,128
                ],
                "predict_layers": [
                    2048,2048,2048,                     
                    1024,1024,1024,                     
                    512,512,512,                     
                    256,256,256,1
                    ],
                "multi_fields_count": CAT_FEATURE_COUNT * 6,
                "num_cross_layers": 12,
                "cross_layer_projection_dim": 1024,
                "model": "dcn_v2",
            }
        )
    )
    params = get_opts(sys.argv, params)
    set_all_seed(params)
    params.deep_layers= [val * 2 for val in params.deep_layers]
    params.predict_layers= [val * 2 for val in params.predict_layers]
    params.embedding_size = [8, 16, 32, 64, 128]
    logger.info(params)

    model = DCNV2MultiHot(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(params, model, optimizer, load_data, TestCriteoMultihotHandler(params))

    handler.run()

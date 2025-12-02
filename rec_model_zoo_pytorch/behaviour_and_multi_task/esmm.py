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
from deepctr_torch.models import ESMM
from deepctr_torch.inputs import DenseFeat
from easydict import EasyDict as edict

from datasets.aliccp import load_data, TestAliccpHandler, get_spec
from utils.handler import ModelHandler, get_params, get_opts
from utils.common import get_loop_element
from utils.logger import logger


class SlicelessESMM(ESMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        dnn_input = x

        ctr_output = self.ctr_dnn(dnn_input)
        cvr_output = self.cvr_dnn(dnn_input)

        ctr_logit = self.ctr_dnn_final_layer(ctr_output)
        cvr_logit = self.cvr_dnn_final_layer(cvr_output)

        ctr_pred = self.out(ctr_logit)
        cvr_pred = self.out(cvr_logit)

        ctcvr_pred = ctr_pred * cvr_pred  # CTCVR = CTR * CVR

        task_outs = torch.cat([ctr_pred, ctcvr_pred], -1)
        return task_outs


class ESMMHandler(nn.Module):
    def __init__(self, params, spec) -> None:
        super().__init__()
        self.spec = spec
        self.params = params
        self.feature_columns = []
        self.emb_weights = {}

        idx = 0
        total_dims = 0
        for key, vocab_len in self.spec["vocab_length"].items():
            dim = get_loop_element(self.params.embedding_size, idx)
            total_dims += dim
            idx += 1
            feat = DenseFeat(key, dimension=dim)
            self.feature_columns.append(feat)
            generator = torch.Generator().manual_seed(idx)
            self.emb_weights[key] = torch.nn.Embedding.from_pretrained(
                torch.normal(mean=0, std=(2 / 512) ** 0.5, size=(vocab_len + 1, dim), generator=generator)
            ).to(self.params.device)
            
        self.model = SlicelessESMM(
            self.feature_columns,
            tower_dnn_hidden_units=params.tower_dnn_hidden_size,
            task_types=("binary", "binary"),
            task_names=("ctr", "ctcvr"),
            device=params.device,
        )

    def forward(self, features, mode="train"):
        embeddings = []
        for key in self.spec["one_hot_fields"]:
            tmp_emb = self.emb_weights.get(key)
            emb_feats = tmp_emb(features[key])
            embeddings.append(
                torch.reshape(emb_feats, [-1, 1, emb_feats.shape[-1]]).squeeze(dim=1)
            )

        for key in self.spec["multi_hot_fields"]:
            feature_dense = features.get(key)
            feature_dense = torch.where(
                feature_dense == -1, torch.zeros_like(feature_dense), feature_dense
            )
            embeddings.append(torch.sum(self.emb_weights[key](feature_dense), dim=1))

        for key in self.spec["special_fields"]:
            feature_sparse = features.get(key)
            sparse_marsk = (feature_sparse >= 0).to(torch.float32)[..., None]
            feature_sparse = torch.where(
                feature_sparse == -1, torch.zeros_like(feature_sparse), feature_sparse
            )
            sparse_lookup_embedding = (
                self.emb_weights[key](feature_sparse) * sparse_marsk
            )
            embeddings.append(torch.sum(sparse_lookup_embedding, dim=1))

        input_features = torch.concat(embeddings, dim=1)
        res = self.model(input_features)
        if len(res.shape) == 1:
            res.unsqueeze(dim=0)
        return {"ctr": res[:, 0], "ctcvr": res[:, 1]}

    def loss(self, pred, labels):
        epsilon = 1e-7
        # Weight for the click-through rate (CTR) loss component
        click_weight = 0.14
        # Weight for the conversion rate (CVR) loss component
        conversion_weight = 0.023
        # Weight for the CTR task in the combined loss function
        y_ctr = pred["ctr"]
        y_ctcvr = pred["ctcvr"]

        ctr_loss = -(1 - click_weight) / click_weight * labels["y"] * torch.log(
            y_ctr + epsilon
        ) - (1 - labels["y"]) * torch.log(1 - y_ctr + epsilon)
        ctr_loss = torch.mean(ctr_loss)

        ctcvr_loss = -(1 - conversion_weight) / conversion_weight * labels[
            "z"
        ] * torch.log(y_ctcvr + epsilon) - (1 - labels["z"]) * torch.log(
            1 - y_ctcvr + epsilon
        )
        ctcvr_loss = torch.mean(ctcvr_loss)
        return ctr_loss + ctcvr_loss


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "reuse_hash": True,
                "tower_dnn_hidden_size": [
                    2048,
                    2048,
                    2048,
                    2048,
                    1024,
                    1024,
                    1024,
                    1024,
                    512,
                    512,
                    512,
                    512,
                    256,
                    256,
                    256,
                    256,
                    128,
                    128,
                    128,
                    128,
                ],
                "extra_fields": 200,
                "model": "esmm",
            }
        )
    )
    params = get_opts(sys.argv, params)
    params.tower_dnn_hidden_size = [val * 3 for val in params.tower_dnn_hidden_size]
    params.embedding_size = [8, 16, 32, 64, 128]
    logger.info(params)

    # 加载数据
    spec = get_spec(params)
    model = ESMMHandler(params, spec).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params, model, optimizer, load_data, TestAliccpHandler(params, spec)
    )
    handler.run()
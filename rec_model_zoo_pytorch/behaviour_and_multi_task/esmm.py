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

import os
import sys

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
from deepctr_torch.inputs import DenseFeat
from deepctr_torch.models import MMOE
from easydict import EasyDict as edict

from datasets.aliccp import load_data, TestAliccpHandler, get_spec
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed
from utils.common import get_loop_element
from utils.logger import logger


class MyMMOE(MMOE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        dnn_input = x
        # expert dnn
        expert_outs = []
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)  # (bs, num_experts, dim)

        # gate dnn
        mmoe_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), expert_outs)  # (bs, 1, dim)
            mmoe_outs.append(gate_mul_expert.squeeze())

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](mmoe_outs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs


class MMOEHandler(nn.Module):
    def __init__(self, params, spec) -> None:
        super().__init__()
        self.spec = spec
        self.params = params
        self.feature_columns = []

        idx = 0
        total_dims = 0
        self.emb_weights = nn.ModuleDict()
        for key, vocab_len in self.spec["vocab_length"].items():
            dim = get_loop_element(params.embedding_size, idx)
            total_dims += dim
            idx += 1
            self.emb_weights[key] = torch.nn.Embedding.from_pretrained(
                torch.normal(mean=0, std=(2 / 512) ** 0.5, size=(vocab_len + 1, dim)),
                freeze=False,
            ).to(self.params.device)
            feat = DenseFeat(key, dimension=dim)
            self.feature_columns.append(feat)

        self.model = MyMMOE(
            dnn_feature_columns=self.feature_columns,
            num_experts=params.num_experts,
            expert_dnn_hidden_units=params.expert_dnn_hidden_size,
            tower_dnn_hidden_units=params.tower_dnn_hidden_size,
            device=self.params.device,
        )

    def forward(self, features, mode="train"):
        embeddings = []
        for key in self.spec["one_hot_fields"]:
            tmp_emb = self.emb_weights[key](features[key])
            embeddings.append(
                torch.reshape(tmp_emb, [-1, 1, tmp_emb.shape[-1]]).squeeze(dim=1)
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
        res = res.view(-1, 2)
        return {"ctr": res[:, 0], "ctcvr": res[:, 1]}

    def loss(self, pred, labels):
        epsilon = 1e-7
        click_weight = 0.14
        conversion_weight = 0.023
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
                "hash_bits": 32,
                "num_experts": 8,
                "expert_dnn_hidden_size": [
                    2048,
                    2048,
                    2048,
                    1024,
                    1024,
                    1024,
                    512,
                    512,
                    512,
                    256,
                    256,
                    256,
                ],
                "tower_dnn_hidden_size": [
                    1024,
                    1024,
                    1024,
                    512,
                    512,
                    512,
                    256,
                    256,
                    256,
                    128,
                    128,
                    128,
                ],
                "model": "mmoe",
                "extra_fields": 300,
            }
        )
    )

    params = get_opts(sys.argv, params)
    set_all_seed(params)
    params.expert_dnn_hidden_size = [val * 2 for val in params.expert_dnn_hidden_size]
    params.tower_dnn_hidden_size = [val * 2 for val in params.tower_dnn_hidden_size]
    params.embedding_size = [8, 16, 32, 64, 128]
    logger.info(params)

    spec = get_spec(params)
    model = MMOEHandler(params, spec).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params, model, optimizer, load_data, TestAliccpHandler(params, spec)
    )
    handler.run()
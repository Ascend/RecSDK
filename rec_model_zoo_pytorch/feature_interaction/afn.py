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
import torch
import torch.nn as nn
from easydict import EasyDict as edict

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

from datasets.criteo import load_data, TestCriteoHandler
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import AFN

class AFNModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = params["device"]
        self.field_size = params["field_size"]
        self.embed_dim = params.get("embedding_size", 16)
        self.hash_space = max(params.get("vocab_size", 2**18), 2**16)  # 给足够大词表
        self.dense_cols = [f"I{i}" for i in range(1, 14)]  # I1~I13
        self.sparse_cols = [f"C{i}" for i in range(1, 27)]  # C1~C26

        dense_feats = [DenseFeat(c, 1) for c in self.dense_cols]
        sparse_feats = [
            SparseFeat(
                c,
                vocabulary_size=self.hash_space,
                embedding_dim=self.embed_dim,
                use_hash=False,
            )
            for c in self.sparse_cols
        ]
        self.feature_columns = dense_feats + sparse_feats

        self.model = AFN(
            linear_feature_columns=self.feature_columns,
            dnn_feature_columns=self.feature_columns,
            # afn_hidden_units=afn_hidden_units,
            # dnn_hidden_units=dnn_hidden_units,
            # dnn_dropout=0.2,
            l2_reg_linear=1e-5,
            l2_reg_embedding=1e-5,
            task="binary",
            device=self.device,
            seed=2024,
        )

        self.criterion = nn.BCELoss()

    def _split_criteo_tensors(self, features):
        fs = self.field_size
        feat_vals = torch.reshape(features["feat_vals"], (-1, fs)).to(torch.float32)
        feat_ids = torch.reshape(features["feat_ids"], (-1, fs)).to(torch.long)
        dense_tensor = feat_vals[:, :13]
        sparse_tensor = feat_ids[:, -26:]
        return dense_tensor, sparse_tensor

    def forward(self, *args, **kwargs):
        features = args[0] if len(args) >= 1 else kwargs.get("features")

        dense_tensor, sparse_tensor = self._split_criteo_tensors(features)
        X_mat = torch.cat([dense_tensor, sparse_tensor.to(torch.float32)], dim=1)

        preds = self.model(X_mat).squeeze(1)
        return {"ctr": preds}

    def loss(self, pred, labels):
        ctr = pred["ctr"]
        if ctr.dim() == 2 and ctr.size(1) == 1:
            ctr = ctr.squeeze(1)
        labels = labels.to(torch.float32).to(self.device)
        return self.criterion(ctr, labels)

if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "attention_dim": 16,
                "num_heads": 4,
                "vocab_size": 2100000,
                # "embedding_size": 16,
                "deep_layers": [512, 256],
                "predict_layers": [1024, 1024, 512, 256, 1],
                # "learning_rate": 0.001,
                # "device": "cuda" if torch.cuda.is_available() else "cpu",
                # "mode": "train",  # 模式：train, eval, test
                "model": "afn",  # 模型类型
            }
        )
    )

    params = get_opts(sys.argv, params)
    set_all_seed(params)
    model = AFNModel(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params, model, optimizer, load_data, TestCriteoHandler(params)
    )
    handler.run()
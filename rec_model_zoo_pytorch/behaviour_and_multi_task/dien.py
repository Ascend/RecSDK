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

from deepctr_torch.models import DIEN
from deepctr_torch.models.dien import *
from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat
from easydict import EasyDict as edict
import torch
import torch.nn as nn

from datasets.aliccp import load_data, TestAliccpHandler, get_spec
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed
from utils.logger import logger
from utils.self_gru import MyExtractor, MyInterestEvolving

MULTIHOT_MAP = {
    '109_14': '206',
    '110_14': '207',
    '127_14': '216',
    '150_14': '210',
}


class MyDIEN(DIEN):
    def __init__(self,
                 dnn_feature_columns, history_feature_list,
                 gru_type="GRU", use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(256, 128),
                 dnn_activation='relu',
                 att_hidden_units=(64, 16), att_activation="relu", att_weight_normalization=True,
                 l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary',
                 device='cpu', gpus=None):

        self.use_negsampling = use_negsampling
        self.item_features = history_feature_list

        super(MyDIEN, self).__init__(dnn_feature_columns=dnn_feature_columns,
                                     history_feature_list=history_feature_list,
                                     dnn_hidden_units=dnn_hidden_units,
                                     att_hidden_units=att_hidden_units,
                                     device=device)

        self._split_columns()
        input_size = self._compute_interest_dim()
        linear_feature_columns = []

        self.interest_extractor = MyExtractor(
            input_size=input_size,
            use_neg=use_negsampling,
            init_std=init_std,
            device=self.device
        )

        self.interest_evolving = MyInterestEvolving(
            input_size=input_size,
            att_hidden_size=att_hidden_units,
            att_activation=att_activation,
            gru_type=gru_type,
            init_std=init_std,
            device=self.device
        )

        self.to(device)

    def _split_columns(self):
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.dnn_feature_columns)) if len(
            self.dnn_feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), self.dnn_feature_columns)) if len(
            self.dnn_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat),
                   self.dnn_feature_columns)) if len(self.dnn_feature_columns) else []

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.item_features:
                interest_dim += feat.embedding_dim
        return interest_dim


class DIENHandler(nn.Module):
    def __init__(self, params, spec) -> None:
        super().__init__()
        self.spec = spec
        self.params = params
        self.feature_columns = []
        cur_mode = params.mode
        if cur_mode == 'eval':
            cur_mode = 'val'
        elif 'test' in cur_mode:
            cur_mode = 'test'
        else:
            cur_mode = 'test'
        for key in self.spec['one_hot_fields']:
            feat = SparseFeat(key, vocabulary_size=self.spec['vocab_length'][key] + 1,
                embedding_dim=params.embedding_size)
            self.feature_columns.append(feat)
        for key in self.spec['multi_hot_fields']:
            if MULTIHOT_MAP.get(key, None):
                feat = SparseFeat(f'hist_{MULTIHOT_MAP[key]}', vocabulary_size=spec['vocab_length'][key] + 1,
                embedding_dim=params.embedding_size)
            else:
                feat = SparseFeat(f'hist_{key}', vocabulary_size=spec['vocab_length'][key] + 1,
                embedding_dim=params.embedding_size)
            feat = VarLenSparseFeat(feat, maxlen=spec[f'{cur_mode}_max_length'][key], length_name='seq_length')
            self.feature_columns.append(feat)
        for key in self.spec['special_fields']:
            feat = SparseFeat(key, vocabulary_size=self.spec['vocab_length'][key] + 1,
            embedding_dim=params.embedding_size)
            feat = VarLenSparseFeat(feat, maxlen=spec[f'{cur_mode}_max_length'][key], length_name='seq_length')
            self.feature_columns.append(feat)
        self.emb_weights = {}
        self.model = MyDIEN(dnn_feature_columns=self.feature_columns,
                            history_feature_list=['206', '207', '216'],
                            dnn_hidden_units=params.dnn_hidden_size,
                            att_hidden_units=params.att_hidden_size,
                            device=params.device)

    def forward(self, features, mode='train'):
        embeddings = []
        for key in self.spec['one_hot_fields']:
            feature = features.get(key).to(params.device)
            feature = torch.where(feature == -1, torch.zeros_like(feature), feature)
            embeddings.append(feature[:, None])

        add_seq_len = False
        for key in self.spec['multi_hot_fields']:
            feature_dense = features.get(key).to(params.device)
            feature_dense = torch.where(feature_dense == -1, torch.zeros_like(feature_dense), feature_dense)
            embeddings.append(feature_dense)
            if not add_seq_len:
                add_seq_len = True
                embeddings.append(torch.full((feature_dense.shape[0], 1), 50, dtype=torch.long).to(params.device))

        for key in self.spec['special_fields']:
            feature_sparse = features.get(key).to(params.device)
            feature_sparse = torch.where(feature_sparse == -1, torch.zeros_like(feature_sparse), feature_sparse)
            embeddings.append(feature_sparse)

        input_features = torch.concat(embeddings, dim=1)
        return {'ctr': self.model(input_features)}

    def loss(self, pred, labels):
        epsilon = 1e-7
        # Weight for the click-through rate (CTR) loss component
        click_weight = 0.14
        # Weight for the conversion rate (CVR) loss component
        conversion_weight = 0.023
        # Weight for the CTR task in the combined loss function
        y_ctr = pred['ctr']
        ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * torch.log(y_ctr + epsilon) - \
                   (1 - labels['y']) * torch.log(1 - y_ctr + epsilon)
        ctr_loss = torch.mean(ctr_loss)
        return ctr_loss


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "reuse_hash": True,
                "hash_bits": 32,
                "dnn_hidden_size": [
                    1024,
                    1024,
                    1024,
                    1024,
                    256,
                    256,
                    256,
                    256,
                    128,
                    128,
                    128,
                    128,
                    ],
                "att_hidden_size": [
                    1024,
                    1024,
                    1024,
                    1024,
                    256,
                    256,
                    256,
                    256,
                    128,
                    128,
                    128,
                    128,
                    64,
                    64,
                    64,
                    64,
                    16,
                    16,
                    16,
                    16,
                    ],
            "extra_fields": 100,
            "model": "dien",
            }
            )
    )
    params = get_opts(sys.argv, params)
    set_all_seed(params)
    params.dnn_hidden_size = [val * 2 for val in params.dnn_hidden_size]
    params.att_hidden_size = [val * 2 for val in params.att_hidden_size]
    logger.info(params)

    # 加载数据
    spec = get_spec(params)
    model = DIENHandler(params, spec).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    handler = ModelHandler(
        params, model, optimizer, load_data, TestAliccpHandler(params, spec)
    )
    handler.run()
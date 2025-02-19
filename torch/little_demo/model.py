# coding: UTF-8
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

from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, PoolingType
from hybrid_torchrec import HashEmbeddingBagCollection
from dataset import Batch
import torch


class TestModel(torch.nn.Module):
    def __init__(self, table_names, feat_names, embed_dims, num_embeds):
        super().__init__()
        table_configs = []
        for table_name, feat_name, dim, num_embed in zip(
            table_names, feat_names, embed_dims, num_embeds
        ):
            config = EmbeddingBagConfig(
                name=table_name,
                embedding_dim=dim,
                num_embeddings=num_embed,
                feature_names=feat_name,
                pooling=PoolingType.SUM,
            )
            table_configs.append(config)

        self.ebc = HashEmbeddingBagCollection(device="npu", tables=table_configs)
        self.input_dim = sum([len(f) * d for f, d in zip(feat_names, embed_dims)])

    def forward(self, batch: Batch):
        result = self.ebc(batch.sparse_features)
        result: torch.Tensor = result.values()
        loss = result.mean() + result.sum() + result.max() + result.min()
        return loss, resul

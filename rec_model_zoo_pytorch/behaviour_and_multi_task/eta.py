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
from datasets.aliccp import load_data, TestAliccpHandler, get_spec
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed
from utils.common import get_loop_element
from utils.logger import logger


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_heads = params.num_heads
        self.attention_dim = params.embedding_size[-1]

        self.q_linear = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.k_linear = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.v_linear = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.out_linear = nn.Linear(self.attention_dim, self.attention_dim, bias=False)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(batch_size, -1, self.num_heads, q.shape[-1] // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, k.shape[-1] // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, v.shape[-1] // self.num_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores /= torch.sqrt(torch.tensor(query.shape[-1] // self.num_heads))

        paddings = torch.ones_like(scores) * (-(2**32) + 1)
        scores = torch.where(
            torch.tile(torch.reshape(mask, [-1, 1, 1, key.shape[1]]), [1, self.num_heads, 1, 1]), scores, paddings
        )

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.attention_dim)
        output = self.out_linear(context)

        return output


class LoneAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.attention = MultiHeadAttention(params)
        if self.params.reuse_hash:
            self.reuse_hash = torch.randn((self.params.embedding_size[-1], self.params.hash_bits)).to(self.params.device)

        self.relu = nn.ReLU()

    def lsh_hash(self, vec, rand_rotate):
        rotated_vec = torch.matmul(vec, rand_rotate)
        hash_code = self.relu(torch.sign(rotated_vec))
        return hash_code

    def forward(self, target_input, seq_input, mask):
        random_rotations = (
            self.reuse_hash
            if self.params.reuse_hash
            else torch.randn((self.params.embedding_dim, self.params.hash_bits))
        )

        target_hash = self.lsh_hash(target_input, random_rotations)
        seq_hash = self.lsh_hash(seq_input, random_rotations)
        sim_hash = torch.sum(torch.abs(seq_hash - target_hash), dim=-1, keepdim=False)

        min_int = -(2**32) + 1
        paddings = torch.zeros_like(sim_hash) + min_int
        sim_hash = torch.where(torch.reshape(mask, (-1, sim_hash.shape[-1])), sim_hash, paddings)
        _, topk_index = torch.topk(sim_hash, k=self.params.topk, dim=-1, sorted=True)

        topk_emb = torch.gather(seq_input, index=topk_index[..., None].expand(-1, -1, seq_input.size(-1)), dim=1)
        topk_mask = torch.gather(mask, index=topk_index.unsqueeze(dim=1), dim=-1)
        output = self.attention(target_input, topk_emb, topk_emb, topk_mask)
        return output


class ETA(nn.Module):
    def __init__(self, params: edict, spec):
        super().__init__()
        self.params = params
        self.spec = spec
        self.reuse_hash = None

        self.emb_weights = nn.ModuleDict()

        extra_multis = []
        extra_ones = []
        if params.get("extra_multi_hots", None) is not None:
            for i in range(params.extra_multi_hots):
                extra_multis.append(str(1000 + i) + "_14")
                extra_ones.append(str(1000 + i))

        self.target_field_name = ["206", "207", "216", "210"] + extra_ones
        self.target_multi_field = [
            "109_14",
            "110_14",
            "127_14",
            "150_14",
        ] + extra_multis
        self.embedding_init()
 
        self.masks = {}
        self.emb_cats = []

        tmp_layer_list = []
        for i in range(len(self.params.deep_layers)):
            tmp_layer_list.append(
                nn.Linear(
                    in_features=self.deep_layers_num[i],
                    out_features=self.deep_layers_num[i + 1],
                    bias=False,
                )
            )
            tmp_layer_list.append(nn.ReLU())

        self.short_attention = MultiHeadAttention(self.params)
        self.long_attention = LoneAttention(self.params)
        self.deep_layers = nn.Sequential(*tmp_layer_list)
        self.out_layer = nn.Linear(in_features=self.deep_layers_num[-1], out_features=1)

    def embedding_init(self):
        idx = 0
        total_dims = 0
        created_fields = []
        for i, _ in enumerate(self.target_field_name):
            one_hot_target = self.target_field_name[i]
            multi_hot_target = self.target_multi_field[i]
            created_fields.extend([one_hot_target, multi_hot_target])
            dim = self.params.embedding_size[-1]
            idx += 1
            total_dims += dim * 2
            self.emb_weights[one_hot_target] = torch.nn.Embedding.from_pretrained(
                torch.normal(
                    mean=0,
                    std=(2 / 512) ** 0.5,
                    size=(self.spec["vocab_length"][one_hot_target] + 1, dim),
                ),
                freeze=False,
            ).to(self.params.device)
            self.emb_weights[multi_hot_target] = torch.nn.Embedding.from_pretrained(
                torch.normal(
                    mean=0,
                    std=(2 / 512) ** 0.5,
                    size=(self.spec["vocab_length"][multi_hot_target] + 1, dim),
                ),
                freeze=False,
            ).to(self.params.device)

        total_dims += len(self.target_field_name) * self.params.embedding_size[-1]

        for key, vocab_len in self.spec["vocab_length"].items():
            if key not in created_fields:
                dims = get_loop_element(params.embedding_size, idx)
                idx += 1
                total_dims += dims
                self.emb_weights[key] = torch.nn.Embedding.from_pretrained(
                    torch.normal(
                        mean=0, std=(2 / 512) ** 0.5, size=(vocab_len + 1, dims)
                    ),
                    freeze=False,
                ).to(self.params.device)

        self.deep_layers_num = [total_dims]
        self.deep_layers_num.extend(self.params.deep_layers)

    def long_emb_padding(self, field_name, embeddings, masks):
        dense_embedding = embeddings.get(field_name)
        dense_mask = masks.get(field_name)
        paddings = [0, 0, 0, params.max_seq_len]
        mask_paddings = [0, params.max_seq_len]
        dense_embedding = F.pad(dense_embedding, paddings, mode="constant", value=0)
        dense_mask = F.pad(dense_mask, mask_paddings, mode="constant", value=0)

        return (
            dense_embedding[:, : params.topk],
            dense_embedding[:, : params.max_seq_len],
            dense_mask[:, :, : params.topk],
            dense_mask[:, :, : params.max_seq_len],
        )

    def embedding_preprocess(self, features):
        embeddings = {}
        for key in self.spec["one_hot_fields"]:
            tmp_emb = self.emb_weights[key]
            embeddings[key] = tmp_emb(features[key])
            embeddings[key] = torch.reshape(
                embeddings[key], [-1, 1, embeddings[key].shape[-1]]
            )

        for key in self.spec["multi_hot_fields"]:
            feature_dense = features.get(key)
            self.masks[key] = (feature_dense >= 0)[:, None]
            feature_dense = torch.where(
                feature_dense == -1, torch.zeros_like(feature_dense), feature_dense
            )
            embeddings[key] = self.emb_weights[key](feature_dense)

        for key in self.spec["special_fields"]:
            feature_sparse = features.get(key)
            sparse_marsk = (feature_sparse >= 0).to(torch.float32)[..., None]
            feature_sparse = torch.where(
                feature_sparse == -1, torch.zeros_like(feature_sparse), feature_sparse
            )
            sparse_lookup_embedding = (
                self.emb_weights[key](feature_sparse) * sparse_marsk
            )
            embeddings[key] = torch.sum(sparse_lookup_embedding, dim=1)[:, None]
        return embeddings

    def forward(self, features, mode="train"):
        embeddings = self.embedding_preprocess(features)

        self.emb_cats = [
            self.long_emb_padding(field, embeddings, self.masks)
            for field in self.target_multi_field
        ]
        short_attentions_arr = []
        for _, (emb_cat, target_name) in enumerate(
            zip(self.emb_cats, self.target_field_name)
        ):
            emb_target = embeddings.get(target_name)
            emb_short = emb_cat[0]
            mask_short = emb_cat[2]
            short_attentions_arr.append(
                self.short_attention(emb_target, emb_short, emb_short, mask_short)
            )

        long_attentions_arr = []
        for _, (emb_cat, target_name) in enumerate(
            zip(self.emb_cats, self.target_field_name)
        ):
            emb_target = embeddings.get(target_name)
            emb_long = emb_cat[1]
            mask_long = emb_cat[3]
            long_attentions_arr.append(
                self.long_attention(emb_target, emb_long, mask_long)
            )

        embedding = torch.concat(
            [embeddings.get(field_name) for field_name in self.spec["one_hot_fields"]]
            + [embeddings.get(field_name) for field_name in self.spec["special_fields"]]
            + short_attentions_arr
            + long_attentions_arr,
            dim=-1,
        )

        x_deep = torch.reshape(embedding, [-1, embedding.shape[-1]])

        x_deep = self.deep_layers(x_deep)
        y_deep = self.out_layer(x_deep)

        y = torch.reshape(y_deep, shape=[-1])
        pred = torch.sigmoid(y)

        return {"ctr": pred}

    def loss(self, pred, labels):
        pred_ctr = pred["ctr"]
        y = labels["y"]
        epsilon = 1e-7
        click_weight = 0.14

        ctr_loss = -(1 - click_weight) / click_weight * y * torch.log(
            pred_ctr + epsilon
        ) - (1 - y) * torch.log(1 - pred_ctr + epsilon)

        loss = torch.mean(ctr_loss)
        return loss


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "num_heads": 4,
                "deep_layers": [
                    2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 
                    256, 256, 266, 256, 128, 128, 128, 128,
                ],
                "reuse_hash": True,
                "hash_bits": 128,
                "topk": 8,
                "extra_fields": 15,
                "model": "eta",
            }
        )
    )
    params = get_opts(sys.argv, params)
    set_all_seed(params)
    params.deep_layers = [val * 6 for val in params.deep_layers]
    params.embedding_size = [8, 16, 32, 64, 128]
    logger.info(params)

    spec = get_spec(params)
    model = ETA(params, spec).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(
        params, model, optimizer, load_data, TestAliccpHandler(params, spec)
    )
    handler.run()
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
import torch
import sys
import getopt
import os

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

from easydict import EasyDict as edict
import torch.nn as nn

# from datasets.aliccp import load_data,test_qps
from datasets.aliccp import load_data, TestAliccpHandler, get_spec
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_normal_like_tf(param: torch.nn.Parameter, std: float = (2 / 512) ** 0.5):
    """TF used tf.random_normal_initializer(stddev=sqrt(2/512) = 1/16 = 0.0625)."""
    with torch.no_grad():
        param.normal_(mean=0.0, std=std)


def _make_embedding(num_embeddings: int, embedding_dim: int) -> nn.Embedding:
    emb = nn.Embedding(
        num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim, padding_idx=0
    )
    _init_normal_like_tf(emb.weight)  # match TF normal init
    return emb


class CANConfig:
    embedding_size: int = 16
    deep_layers: Tuple[int, ...] = (512, 256, 128, 64)
    weight_emb_w: List[List[int]] = None  # e.g. [[12, 6], [6, 4]]
    weight_emb_b: List[int] = None  # e.g. [0, 0]
    orders: int = 3
    click_weight: float = 0.14
    epsilon: float = 1e-7

    def __post_init__(self):
        if self.weight_emb_w is None:
            self.weight_emb_w = [[12, 6], [6, 4]]
        if self.weight_emb_b is None:
            self.weight_emb_b = [0, 0]


class CANModel(nn.Module):
    def __init__(self, params, spec):
        super().__init__()
        self.params = params
        self.spec = spec
        self.embedding_size = self.params["embedding_size"]
        self.deep_layers = self.params["deep_layers"]
        self.weight_emb_w = [[12, 6], [6, 4]]
        self.weight_emb_b = [0, 0]
        self.orders: int = 3
        self.click_weight: float = 0.14
        self.epsilon: float = 1e-7

        # Field groups (with sane defaults if not provided)
        self.one_hot_fields: List[str] = list(
            spec.get(
                "one_hot_fields",
                [
                    "101",
                    "121",
                    "122",
                    "124",
                    "125",
                    "126",
                    "127",
                    "128",
                    "129",
                    "205",
                    "206",
                    "207",
                    "216",
                    "508",
                    "509",
                    "702",
                    "301",
                ],
            )
        )
        self.multi_hot_fields: List[str] = list(
            spec.get(
                "multi_hot_fields",
                ["109_14", "110_14", "127_14", "150_14", "210", "853"],
            )
        )
        self.special_fields: List[str] = list(spec.get("special_fields", []))
        self.category_fields: List[str] = list(
            spec.get(
                "category_fields",
                ["101", "121", "122", "124", "125", "126", "127", "128", "129"],
            )
        )
        self.coaction_pairs: List[List[str]] = list(
            spec.get(
                "coaction_pairs",
                [
                    ["206", "109_14"],
                    ["207", "110_14"],
                    ["216", "127_14"],
                    ["210", "150_14"],
                ],
            )
        )

        vocab_length: Dict[str, int] = spec["vocab_length"]

        # Base embeddings for all fields that appear in the top-level concat (embedding_size)
        all_base_fields = (
            set(self.one_hot_fields)
            | set(self.multi_hot_fields)
            | set(self.special_fields)
        )
        self.base_emb = nn.ModuleDict(
            {
                f: _make_embedding(vocab_length[f], self.embedding_size)
                for f in all_base_fields
            }
        )

        # Category embeddings (first-stage coaction dimension = weight_emb_w[0][0])
        self.cat_dim = int(self.weight_emb_w[0][0])
        self.cate_weights = nn.ModuleDict(
            {
                f: _make_embedding(vocab_length[f], self.cat_dim)
                for f in self.category_fields
            }
        )

        # Coaction dynamic weights: per-field trainable lookup tables
        # target: produces vector sliced into multiple (W, b) blocks
        self.weight_emb_dim = sum(w[0] * w[1] for w in self.weight_emb_w) + sum(
            self.weight_emb_b
        )
        self.target_coaction = nn.ModuleDict(
            {
                tgt: _make_embedding(vocab_length[tgt], self.weight_emb_dim)
                for (tgt, _his) in self.coaction_pairs
            }
        )
        # history/sequence: projects ids -> cat_dim
        self.his_coaction = nn.ModuleDict(
            {
                his: _make_embedding(vocab_length[his], self.cat_dim)
                for (_tgt, his) in self.coaction_pairs
            }
        )

        # MLP stack + PReLUs (alpha init = 0 to mirror TF p_re_lu)
        # todo: 换了一个精确预估值
        # dims = [self._flat_out_dim_hint()] + list(self.deep_layers)
        dims = [self._calculate_flat_out_dim()] + list(self.deep_layers)
        self.mlp = nn.ModuleList()
        self.prelu = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(in_d, out_d)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.mlp.append(lin)
            pre = nn.PReLU(num_parameters=out_d, init=0.0)
            self.prelu.append(pre)

        # Output head
        self.out = nn.Linear(self.deep_layers[-1], 1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # The TF BN is effectively a no-op; we keep parity via Identity
        self.bn = nn.Identity()

    def forward(self, features, mode="train"):
        # 1) Base embedding layer (None x 1 x E per field)
        base_embeddings = self._build_embedding_layer(features)

        # 2) Coaction layer (None x 1 x ?)
        coaction_tensor = self._build_coaction_layer(features)
        # print(f'coaction_tensor : {coaction_tensor.shape}')
        # 3) Final concat over last dim
        emb_list = [
            *[base_embeddings[f] for f in self.one_hot_fields if f in base_embeddings],
            *[
                base_embeddings[f]
                for f in self.multi_hot_fields
                if f in base_embeddings
            ],
            *[base_embeddings[f] for f in self.special_fields if f in base_embeddings],
            coaction_tensor,
        ]
        embedding = torch.cat(emb_list, dim=2)  # [B, 1, D]
        # 4) MLP
        x = embedding.reshape(embedding.size(0), -1)
        x = self.bn(x)  # no-op, kept for structural parity
        # print(f'x.shape : {x.shape}')
        for lin, act in zip(self.mlp, self.prelu):
            # print(f'lin.weight.shape : {lin.weight.shape}')
            x = act(lin(x))

        # 5) Output
        logit = self.out(x).squeeze(-1)
        prob = torch.sigmoid(logit)
        return {"prob": prob}

    # todo: 传进来的prob是dict，不是tensor
    def loss(
        self, outputs: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        prob = outputs["prob"]  # 取出真正的预测值
        y = y["y"]
        eps = self.epsilon
        cw = self.click_weight
        # y, prob are shape [B]
        # print(f'y : {type(y)}, prob : {prob}')
        loss = -(1 - cw) / cw * y * torch.log(prob + eps) - (1 - y) * torch.log(
            1 - prob + eps
        )
        return loss.mean()

    # --------- internals matching TF math ---------

    def _build_embedding_layer(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        B = None
        for f in self.one_hot_fields:
            if f not in self.base_emb:  # skip if not present
                continue
            x = features[f].long()  # [B]
            e = self.base_emb[f](x).unsqueeze(1)  # [B,1,E]
            out[f] = e
            B = x.size(0)
        for f in self.multi_hot_fields:
            if f not in self.base_emb:
                continue
            idx = features[f].long()  # [B,L] with -1 pads
            e = self._embedding_lookup_sparse_sum(self.base_emb[f], idx)  # [B,E]
            out[f] = e.unsqueeze(1)  # [B,1,E]
            if B is None:
                B = idx.size(0)
        # If some special fields should be included at top-level (often one-hots)
        for f in self.special_fields:
            if f not in self.base_emb or f in out:
                continue
            x = features[f]
            if x.dim() == 1:
                e = self.base_emb[f](x.long()).unsqueeze(1)
            else:
                e = self._embedding_lookup_sparse_sum(
                    self.base_emb[f], x.long()
                ).unsqueeze(1)
            out[f] = e
        return out

    def _build_coaction_layer(self, features) -> torch.Tensor:
        params = self.params
        orders = self.orders

        # Category block (9 fields -> [B, 9, cat_dim])
        cat_feats = []
        for f in self.category_fields:
            x = features[f].long()  # [B]
            cat_feats.append(self.cate_weights[f](x).unsqueeze(1))  # [B,1,cat_dim]
        category_feat = torch.cat(cat_feats, dim=1)  # [B,9,cat_dim]
        category_orders = [category_feat ** (i + 1) for i in range(orders)]

        coaction_parts: List[torch.Tensor] = []
        i = 0
        for tgt, his in self.coaction_pairs:
            # 1) Target dynamic weights vector -> sliced into (W, b) blocks
            tgt_idx = features[tgt]
            if tgt in self.special_fields and tgt_idx.dim() == 2:
                tgt_vec = self._embedding_lookup_sparse_sum(
                    self.target_coaction[tgt], tgt_idx.long()
                )  # [B, D*]
            else:
                tgt_vec = (
                    self.target_coaction[tgt](tgt_idx.long()).squeeze(1)
                    if tgt_idx.dim() == 2
                    else self.target_coaction[tgt](tgt_idx.long())
                )  # [B, D*]

            weights, biases = self._slice_dynamic_params(tgt_vec)

            # 2) History embeddings (sequence) and mask
            his_idx = features[his].long()  # [B,T] with -1 pads
            mask = (his_idx >= 0).unsqueeze(-1)  # [B,T,1]
            his_idx = his_idx.clamp_min(0)
            his_emb = self.his_coaction[his](his_idx)  # [B,T,cat_dim]

            # (a) Sequential branch
            seq_outs = []
            h_list = [his_emb ** (i + 1) for i in range(orders)]
            for h in h_list:
                for W, b in zip(weights, biases):
                    # h: [B,T,in], W: [B,in,out]
                    h = torch.matmul(h, W)
                    if b is not None:
                        h = h + b  # [B,1,out] broadcast over T
                    if W is not weights[-1]:
                        h = torch.tanh(h)
                    # print(f'h.shape : {h.shape}')

                    seq_outs.append(h)
            seq_cat = torch.cat(seq_outs, dim=-1)
            seq_cat = torch.where(mask, seq_cat, torch.zeros_like(seq_cat))
            seq_out = seq_cat.sum(dim=1, keepdim=True)  # [B,1,?]
            coaction_parts.append(seq_out)

            # (b) Non-sequential branch (category_orders)
            nseq_outs = []
            for h in category_orders:  # [B,9,cat_dim]
                x = h
                for W, b in zip(weights, biases):
                    x = torch.matmul(x, W)  # [B,9,out]
                    if b is not None:
                        x = x + b  # [B,1,out] broadcast over 9
                    if W is not weights[-1]:
                        x = torch.tanh(x)
                    # todo: 这里是不是少缩进了
                    nseq_outs.append(x)
            nseq_cat = torch.cat(nseq_outs, dim=-1)  # [B,9,?]
            nseq_out = nseq_cat.reshape(nseq_cat.size(0), 1, -1)  # [B,1,9*?]
            # 需要和前面一样sum吗？
            coaction_parts.append(nseq_out)
            # print(f"  Pair {i}: seq_out.shape = {seq_out.shape}, nseq_out.shape = {nseq_out.shape}")
            i += 1

        # Final coaction concat along last dim
        return torch.cat(coaction_parts, dim=2)  # [B,1,?]

    # --------- utilities ---------

    def _embedding_lookup_sparse_sum(
        self, emb: nn.Embedding, idx: torch.Tensor
    ) -> torch.Tensor:
        """Sum combiner over last dimension; -1 is treated as padding (mapped to 0)."""
        if idx.dim() != 2:
            raise ValueError("Expected [B,L] for sparse/multi-hot fields")
        idx = idx.clamp_min(0)
        e = emb(idx)  # [B,L,E]
        return e.sum(dim=1)  # [B,E]

    def _slice_dynamic_params(
        self, vec: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Split per-sample target vector into weight/bias blocks.
        Returns lists of W: [B,in,out] and b: [B,1,out or None].
        """
        B = vec.size(0)
        weights: List[torch.Tensor] = []
        biases: List[torch.Tensor] = []
        i = 0
        for (in_d, out_d), b_dim in zip(self.weight_emb_w, self.weight_emb_b):
            w_sz = in_d * out_d
            W = vec[:, i : i + w_sz].reshape(B, in_d, out_d)
            i += w_sz
            weights.append(W)
            if b_dim and b_dim > 0:
                b = vec[:, i : i + b_dim].reshape(B, 1, b_dim)
                i += b_dim
                biases.append(b)
            else:
                biases.append(None)
        return weights, biases

    def _flat_out_dim_hint(self) -> int:
        """Only used to seed Linear layer shapes. In PyTorch we flatten dynamically anyway, but
        we build with a conservative hint based on the TF formula to keep shapes stable.
        """
        # Conservative lower bound: (count of base fields) * E  +  coaction glue
        n_base = len(
            set(self.one_hot_fields)
            | set(self.multi_hot_fields)
            | set(self.special_fields)
        )
        base_dim = n_base * self.embedding_size
        # Coaction: two outputs per pair (seq + non-seq). We cannot know exact dims without runtime shapes,
        # so we over-approximate using category size (9) and W shapes. We ensure Linear will accept any
        # larger dynamic flatten by rebuilding at first forward if needed; but to keep it simple here, we
        # return a typical value used in the TF code path.
        # The TF graph used: 23*E + 4*orders*sum(w[1]) * 10
        add = 4 * self.orders * sum(w[1] for w in self.weight_emb_w) * 10
        return max(base_dim + add, 128)  # never below 128 to keep MLP sane

    # 位于 CANModel 类内部
    def _calculate_flat_out_dim(self) -> int:
        """Precisely calculate the flattened output dimension before the MLP."""

        # --- 最终修正 ---
        # 使用 set 并集来获取所有唯一的 base field 名称，
        # 这完美地模拟了 _build_embedding_layer 中的去重逻辑。
        unique_base_fields = (
            set(self.one_hot_fields)
            | set(self.multi_hot_fields)
            | set(self.special_fields)
        )
        n_base_fields = len(unique_base_fields)

        base_dim = n_base_fields * self.embedding_size

        # coaction_dim 的计算保持不变
        coaction_dim = 0
        mlp_out_dim_sum = self.orders * sum(w[1] for w in self.weight_emb_w)

        for _ in self.coaction_pairs:
            coaction_dim += mlp_out_dim_sum
            num_category_fields = len(self.category_fields)
            coaction_dim += num_category_fields * mlp_out_dim_sum
        print(f"base_dim : {base_dim}. coaction_dim : {coaction_dim}")

        return base_dim + coaction_dim

    # --------- weight porting (TF -> Torch) ---------

    def load_from_npz(self, npz_path: str):
        """Load parameters from an .npz exported from TF (see stub below for exporter).
        The .npz should contain arrays named after TF variables. You may need to map names
        (e.g., '101_emb_wgts:0' -> 'base_emb.101.weight', etc.).
        """
        arrays = (
            dict(torch.load(npz_path, map_location="cpu"))
            if npz_path.endswith(".pt")
            else dict(__import__("numpy").load(npz_path))
        )

        # Example mappings (adjust to your TF variable names):
        with torch.no_grad():
            # Base embeddings
            for f, emb in self.base_emb.items():
                for k in [f + "_emb_wgts", f + "_emb_wgts:0"]:
                    if k in arrays:
                        w = arrays[k]
                        assert w.shape == tuple(
                            emb.weight.shape
                        ), f"Shape mismatch for {f}: {w.shape} vs {tuple(emb.weight.shape)}"
                        emb.weight.copy_(torch.from_numpy(w))
                        break

            # Category embeddings
            for f, emb in self.cate_weights.items():
                for k in [f + "_cate_emb_wgts", f + "_cate_emb_wgts:0"]:
                    if k in arrays:
                        emb.weight.copy_(torch.from_numpy(arrays[k]))
                        break

            # Coaction tables
            for tgt, his in self.coaction_pairs:
                key = f"target_coaction_emb_weight_{tgt}"
                for k in [key, key + ":0"]:
                    if k in arrays:
                        self.target_coaction[tgt].weight.copy_(
                            torch.from_numpy(arrays[k])
                        )
                        break
                key = f"his_coaction_emb_weight_{his}"
                for k in [key, key + ":0"]:
                    if k in arrays:
                        self.his_coaction[his].weight.copy_(torch.from_numpy(arrays[k]))
                        break

            # MLP layers (you will likely need to map TF contrib fully_connected names)
            # E.g., 'MLP-layer/mlp0/weights', 'MLP-layer/mlp0/biases', ... , 'CAN-out/can_out/weights'
            # This is left as a placeholder as exact TF var names depend on your graph.
            pass


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "max_seq_len": 50,
                "attention_dim": 16,
                "num_heads": 4,
                "deep_layers": [512, 256, 128, 64],
                "reuse_hash": True,
                "hash_bits": 32,
                "topk": 16,
                "model": "can2",
            }
        )
    )
    params = get_opts(sys.argv, params)
    set_all_seed(params)
    spec = get_spec(params)
    # print(spec)
    model = CANModel(params, spec).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    # todo
    handler = ModelHandler(
        params, model, optimizer, load_data, TestAliccpHandler(params, spec)
    )
    handler.run()

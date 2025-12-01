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
import copy
import random
import sys
import os

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict

from datasets.ml_1m import load_data, data_partition
from utils.handler import ModelHandler, get_params, get_opts
from utils.logger import logger


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, embedding_size, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec(torch.nn.Module):
    def __init__(self, params, spec):
        super(SASRec, self).__init__()
        self.params = params
        self.spec = spec
        user_num = spec["user_num"]
        item_num = spec["item_num"]

        self.user_num = user_num
        self.item_num = item_num
        self.dev = params.device
        self.norm_first = params.norm_first

        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, params.embedding_size, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(params.maxlen + 1, params.embedding_size, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=params.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(params.embedding_size, eps=1e-8)

        for _ in range(params.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(params.embedding_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(params.embedding_size, params.num_heads, params.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(params.embedding_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(params.embedding_size, params.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            self.pos_sigmoid = torch.nn.Sigmoid()
            self.neg_sigmoid = torch.nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss()

    def log2feats(self, log_seqs): 
        seqs = self.item_emb(torch.tensor(log_seqs, dtype=torch.long).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= log_seqs != 0
        seqs += self.pos_emb(torch.tensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        layers_length = len(self.attention_layers)
        for i in range(layers_length):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, features, mode="train"):  # for training
        log_seqs = features["seq"]
        pos_seqs = features["pos"]
        neg_seqs = features["neg"]
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.tensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.tensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return [F.sigmoid(pos_logits), F.sigmoid(neg_logits)]

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        preds = self.pos_sigmoid(logits)  # rank same item list for different users

        return preds  # (U, I)

    def loss(self, outputs, labels):
        total_loss = self.bce(outputs[0], torch.ones(outputs[0].shape, device=outputs[0].device))
        total_loss += self.bce(outputs[1], torch.ones(outputs[1].shape, device=outputs[1].device))
        return total_loss


class SASRecHandler(ModelHandler):
    def __init__(self, params, model, optimizer, train_loader, test_loader, val_loader):
        super().__init__(params, model, optimizer, train_loader, test_loader, val_loader)

    def train(self):
        self.model.train()
        train_losses = []
        avg_ndcgs = []
        min_avg_ndcg = 0
        saved_path = self.saved_dir

        for epoch in range(self.params.num_epochs):
            # 训练阶段
            running_loss = 0.0

            running_loss = self.run_train_one_epoch(epoch)

            train_loss = running_loss / (len(self.train_loader) * self.params.num_epochs)
            train_losses.append(train_loss)

            # 验证阶段
            ndcg = self.eval(epoch)

            avg_ndcg = ndcg
            if saved_path is not None and avg_ndcg > min_avg_ndcg:
                min_avg_ndcg = avg_ndcg
                file_path = os.path.join(saved_path, "best_val.pth")
                logger.info(f"saving checkpoint to {file_path}!")
                torch.save(self.model.state_dict(), file_path)

            avg_ndcgs.append(avg_ndcg)

            # 打印统计信息
            logger.info(f"Epoch {epoch+1}/{self.params.num_epochs}")
            logger.info("-" * 50)

    def eval(self, epoch=0):
        self.model.eval()
        dataset = data_partition(self.params.dataset)
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0

        if usernum > 10000:
            users = random.sample(range(1, usernum + 1), 10000)
        else:
            users = range(1, usernum + 1)
        
        profiler = self.get_profiler()
        with profiler as prof:
            for user in users:

                if len(train[user]) < 1 or len(test[user]) < 1:
                    continue

                seq = np.zeros([params.maxlen], dtype=np.int32)
                idx = self.params.maxlen - 1
                seq[idx] = valid[user][0]
                idx -= 1
                for i in reversed(train[user]):
                    seq[idx] = i
                    idx -= 1
                    if idx == -1:
                        break
                rated = set(train[user])
                rated.add(0)
                item_idx = [test[user][0]]
                for _ in range(100):
                    t = np.random.randint(1, itemnum + 1)
                    while t in rated:
                        t = np.random.randint(1, itemnum + 1)
                    item_idx.append(t)

                predictions = -self.model.predict(*[np.array(it) for it in [[user], [seq], item_idx]])
                predictions = predictions[0]  # - for 1st argsort DESC
                rank = predictions.argsort().argsort()[0].item()

                valid_user += 1

                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
                if valid_user % 100 == 0:
                    sys.stdout.flush()
                if ("cuda" in self.params.device or "npu" in self.params.device) and self.params.profiling_mode:
                    prof.step()
        NDCG_10 = NDCG / valid_user
        HR_10 = HT / valid_user
        logger.info("test (NDCG@10: %.4f, HR@10: %.4f)" % (NDCG_10, HR_10))
        return NDCG_10

    def test(self, epoch=0):
        return self.eval(epoch)


if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
                "maxlen": 50,
                "attention_dim": 16,
                "num_heads": 1,
                "dropout_rate": 0.2,
                "batch_size": 128,
                "norm_first": False,
                "num_blocks": 2,
                # 'field_size': 50,
                "field_size": 50,
                "embedding_size": 32,
                "model": "sasrec",
                "dataset": "ml-1m",
            }
        )
    )
    params = get_opts(sys.argv, params)

    # 加载数据
    train_loader, test_loader, val_loader, spec = load_data(params)

    model = SASRec(params, spec).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    handler = SASRecHandler(params, model, optimizer, train_loader, test_loader, val_loader)
    handler.run()

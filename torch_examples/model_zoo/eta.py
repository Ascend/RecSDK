#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
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

import argparse
import glob
import json
import logging
import os
import random
import shutil
import stat
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import pytz

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if torch.cuda.is_available():
    torch.cuda.manual_seed_all2024
else:
    torch.manual_seed(2024)
random.seed(2024)

MODEL_NAME = "ETA"
STD_DEV = (2 / 512) ** 0.5 # 初始化权重标准差
EMBEDDING_FEATURE_NUM = 27 # feature num
TARGET_FIELDS = ["206", "207", "216", "210"] # 4 target fields


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", type=int, default=16, help="Embedding size")
    parser.add_argument("--attention_dim", type=int, default=64, help="")
    parser.add_argument("--num_heads", type=int, default=4, help="")
    parser.add_argument("--short_output_dim", type=int, default=16, help="short attention output dimension")
    parser.add_argument("--max_seq_len", type=int, default=50, help="")
    parser.add_argument("--topk", type=int, default=16, help="")
    parser.add_argument("--deep_layers", type=str, default="512,256,128,64", help="deep layers")
    parser.add_argument("--hash_bits", type=int, default=32, help="")
    parser.add_argument("--reuse_hash", type=bool, default=True, help="")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["Adam", "Adagrad", "GD", "Momentum"], help="")
    parser.add_argument('--early_stop_patience', type=int, default=5, help="")
    parser.add_argument("--data_dir", type=str, default="/aliccp/aliccp_out", help="data dir")
    parser.add_argument("--dt_dir", type=str, default='', help="data dt partition")
    parser.add_argument("--model_dir", type=str, default=f"./", help="code check point dir")
    parser.add_argument("--clear_existing_model", action="store_true", help="")
    parser.add_argument("--task_type", type=str, default="train",
                        choices=["train", "eval", "predict"], help="task type")
    parser.add_argument("--log_level", type=str, default="DEBUG",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Log level")
    parser.add_argument('--epoch_num', type=int, default=1, help="Number of epochs")
    parser.add_argument('--train_batch_num', type=int, default=2000, help="Number of train batchs")
    parser.add_argument('--test_batch_num', type=int, default=10, help="Number of test batchs")
    return parser.parse_args()


def configure_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ETA_trainning_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def json_file_load(json_name: str, json_path: str) -> dict:
    """
    Load a JSON file from the specified path.
    """
    flags = os.O_RDONLY
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    try:
        with os.fdopen(os.open(json_path, flags, modes), "r") as fp:
            json_re = json.load(fp)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{json_name} file not found: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{json_name} contains invalid JSON: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading {json_name} file: {e}") from e

    return json_re


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self._load_hdf5()

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        input_dict = {k: v[idx] for k, v in self.input_sample.items()}
        target_dict = {k: v[idx] for k, v in self.target_sample.items()}
        return input_dict, target_dict

    def _load_hdf5(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            self.input_sample = {}
            self.target_sample = {}
            y = np.array(f["y"])
            z = np.array(f["z"])
            fields = [key for key in f.keys() if key not in ["y", "z"]]
            self.target_sample.update({"y": torch.tensor(y, dtype=torch.float32)})
            self.target_sample.update({"z": torch.tensor(z, dtype=torch.float32)})
            for multi_field in fields:
                self.input_sample.update({multi_field: torch.tensor(np.array(f[multi_field]), dtype=torch.int64)})
        self._length = len(y)
        logger.info(f"LOAD {self.hdf5_path} success")


class TorchDataSet(ConcatDataset):
    def __init__(self, files):
        datasets = [HDF5Dataset(fp) for fp in files]
        super().__init__(datasets)


class ETA(nn.Module):
    def __init__(self, spec, params):
        super(ETA, self).__init__()
        self.spec = spec
        self.params = params

        # embedding layers
        self.emb_weights = nn.ModuleDict()
        for key, vocab_len in spec["vocab_length"].items():
            self.emb_weights[key] = nn.Embedding(vocab_len + 1, params.embedding_size)
            nn.init.normal_(self.emb_weights[key].weight, std=STD_DEV)

        self.hash_weights = nn.Parameter(
            torch.randn(params.embedding_size, params.hash_bits),
            requires_grad=False
        )

        # attention paramters
        self.short_attentions = nn.ModuleList()
        self.long_attentions = nn.ModuleList()
        for _ in range(len(TARGET_FIELDS)): # 4 target fields
            self.short_attentions.append(ShortAttention(params))
            self.long_attentions.append(LongAttention(self.hash_weights, params))

        # mlp layer
        self.mlp = nn.Sequential()
        deep_layers = list(map(int, params.deep_layers.strip().split(",")))
        input_dim = EMBEDDING_FEATURE_NUM * params.embedding_size
        for i, dim in enumerate(deep_layers):
            self.mlp.add_module(f'mlp{i}', nn.Linear(input_dim, dim))
            self.mlp.add_module(f'relu{i}', nn.ReLU())
            input_dim = dim

        # output layer
        self.output = nn.Linear(input_dim, 1)

    def embedding_lookup_sparse(self, params: nn.Embedding, ids: torch.Tensor, combiner: str):
        mask = (ids >= 0).float().unsqueeze(-1)
        ids = ids.clone()
        ids[ids == -1] = 0
        embedding = params(ids) * mask
        if combiner == "sum":
            return embedding.sum(dim=1)
        elif combiner == "mean":
            return embedding.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-7)
        else:
            raise ValueError("combiner only supoort 'sum', 'mean'")

    def forward(self, features):
        # embedding
        embeddings = {}
        masks = {}
        for key in self.spec["one_hot_fields"]:
            embeddings[key] = self.emb_weights[key](features[key]).unsqueeze(1)
        for key in self.spec["multi_hot_fields"]:
            feat = features[key]
            masks[key] = (feat >= 0).bool().unsqueeze(1)
            feat = feat.clone()
            feat[feat == -1] = 0
            embeddings[key] = self.emb_weights[key](feat)
        for key in self.spec["special_fields"]:
            embeddings[key] = self.embedding_lookup_sparse(
                self.emb_weights[key], features[key], combiner="sum"
            ).unsqueeze(1)

        def long_emb_cat(field_name):
            dense_embedding = embeddings.get(field_name)
            dense_mask = masks.get(field_name)
            if dense_embedding is None or dense_mask is None:
                raise ValueError(f"Field {field_name} not found in embeddings or masks")
            padded_embedding = F.pad(dense_embedding, (0, 0, 0, self.params.max_seq_len), "constant", 0)
            padded_mask = F.pad(dense_mask, (0, self.params.max_seq_len), "constant", False)
            return (
                padded_embedding[:, :self.params.topk, :],
                padded_embedding[:, :self.params.max_seq_len, :],
                padded_mask[:, :, :self.params.topk],
                padded_mask[:, :, :self.params.max_seq_len]
            )
        emb_cats = [long_emb_cat(field) for field in self.spec["multi_hot_fields"]]
        target_fields = TARGET_FIELDS

        short_attns = []
        long_attns = []

        for i, (emb_cat, target) in enumerate(zip(emb_cats, target_fields)):
            emb_target = embeddings[target]
            # short-Attention:topk embeddings and masks
            emb_short = emb_cat[0]
            mask_short = emb_cat[2]
            short_attns.append(self.short_attentions[i](emb_target, emb_short, mask_short))
            # long-Attention: max_seq_len embeddings and masks
            emb_long = emb_cat[1]
            mask_long = emb_cat[3]
            long_attns.append(self.long_attentions[i](emb_target, emb_long, mask_long))

        # concat all embeddings
        all_embs = []
        for field in self.spec["one_hot_fields"] + self.spec["special_fields"]:
            # 统一嵌入张量维度，去除冗余维度。
            if embeddings[field].dim() > 3:
                all_embs.append(embeddings[field].squeeze(1))
            else:
                all_embs.append(embeddings[field])
        all_embs += short_attns + long_attns
        all_embs = torch.cat(all_embs, dim=1)
        # 统一嵌入张量维度，便于后续拼接。
        if all_embs.dim() >= 2:
            concat_emb = all_embs.squeeze(1)
        else:
            concat_emb = all_embs

        concat_emb = concat_emb.reshape(concat_emb.size(0), -1)
        # mlp and outputs
        mlp_out = self.mlp(concat_emb)
        logits = self.output(mlp_out).squeeze()
        pred = torch.sigmoid(logits)
        return pred, logits

    # click_weight is 0.14
    def build_loss(self, pred, labels, click_weight=0.14, epsilon=1e-7):
        if pred.shape != labels.shape:
            raise ValueError(f"pred and labels must be the same shape. "
                             f"pred shape: {pred.shape}, labels shape: {labels.shape}")
        pred = torch.clamp(pred, min=epsilon, max=1 - epsilon)
        loss = - (1 - click_weight) / click_weight * labels * torch.log(pred) - (1 - labels) * torch.log(1 - pred)
        return loss.mean()

    def build_optimizer(self):
        if self.params.optimizer == "Adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.params.learning_rate,
                betas=[0.9, 0.999], eps=1e-8
            )
        elif self.params.optimizer == "Adagrad":
            optimizer = optim.Adagrad(
                params=self.parameters(),
                lr=self.params.learning_rate,
                initial_accumulator_value=1e-6
            )
        elif self.params.optimizer == "Momentum":
            optimizer = optim.SGD(
                params=self.parameters(),
                lr=self.params.learning_rate,
                momentum=0.95
            )
        elif self.params.optimizer == "SGD":
            optimizer = optim.SGD(
                params=self.parameters(),
                lr=self.params.learning_rate,
            )
        else:
            raise ValueError("Unsupported optimizer type: {}".format(args.optimizer))
        return optimizer


class ShortAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.attention_dim = params.attention_dim
        self.num_heads = params.num_heads
        self.key_dim = self.attention_dim // self.num_heads

        self.q_fc = nn.Linear(params.embedding_size, self.attention_dim)
        self.k_fc = nn.Linear(params.embedding_size, self.attention_dim)
        self.v_fc = nn.Linear(params.embedding_size, self.attention_dim)
        self.o_fc = nn.Linear(self.attention_dim, params.short_output_dim)

    def forward(self, target_input, seq_input, mask):
        # target_input:[B, 1, E], seq_input:[B, S, E], mask:[B, 1, S] B = target_input.size(0)
        b, tgt_len = target_input.shape[:2]
        b, seq_len = seq_input.shape[:2]
        query = self.q_fc(target_input) # [B, 1, A]
        key = self.k_fc(seq_input) # [B, S, A]
        value = self.v_fc(seq_input) # [B, S, A]

        # split heads
        query = query.view(b, tgt_len, self.num_heads, self.key_dim).permute(0, 2, 1, 3) # [B, Heads, 1, key_dim]
        key = key.view(b, seq_len, self.num_heads, self.key_dim).permute(0, 2, 3, 1) # [B, Heads, key_dim, S]
        value = value.view(b, seq_len, self.num_heads, self.key_dim).permute(0, 2, 1, 3) # [B, Heads, S, key_dim]

        # scaled dot-product attention
        scores = torch.matmul(query, key) / (self.key_dim ** 0.5) # [B, Heads, 1, S]
        scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, value) # [B, Heads, 1, key_dim]
        output = output.permute(0, 2, 1, 3)
        b, lens, heads, dims = output.shape
        output = output.reshape(b * lens, 1, heads * dims)
        output = self.o_fc(output)
        return output


class LongAttention(nn.Module):
    def __init__(self, hash_weights, params):
        super().__init__()
        self.hash_weights = hash_weights
        self.reuse_hash = params.reuse_hash
        self.topk = params.topk
        self.short_attn = ShortAttention(params)

    def lsh_hash(self, vecs, hash_weights):
        rotated_vecs = torch.matmul(vecs, hash_weights)
        return (rotated_vecs > 0).float()

    def forward(self, target_input, seq_input, mask):
        # target_input:[B, 1, E], seq_input:[B, S, E], mask:[B, 1, S]
        b, s, e = seq_input.shape
        hash_weights = self.hash_weights if self.reuse_hash else nn.Parameter(
            torch.randn(e, self.hash_weights.size(-1)),
            requires_grad=False
        ) # [E, hash_bits]

        target_hash = self.lsh_hash(target_input.squeeze(), hash_weights) # [B, hash_bits]
        seq_hash = self.lsh_hash(seq_input, hash_weights) # [B, S, hash_bits]
        # calculate similarity(hamming distance)
        hash_sim = - torch.sum(torch.abs(seq_hash - target_hash.unsqueeze(1)), dim=-1) # [B, S]
        hash_sim = hash_sim.masked_fill(~mask.squeeze(1), float('-inf'))

        # select topk
        _, topk_idx = torch.topk(hash_sim, self.topk, dim=-1)  # [B, topk]

        topk_seq = torch.gather(
            seq_input,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, e)
        )

        topk_idx = topk_idx.unsqueeze(1)
        topk_mask = torch.gather(mask, 2, topk_idx)

        # apply short attention to topk
        output = self.short_attn(target_input, topk_seq, topk_mask)
        return output


def train(model: ETA, train_loader, eval_loader, device, args):
    optimizer = model.build_optimizer()
    best_eval_loss = float('inf')
    no_improve_epochs = 0

    logger.info("Starting model trainning...")
    logger.info(f"Device: {device}")
    logger.info(f"Trainning config: {json.dumps(vars(args), indent=2)}")

    for epoch in range(args.epoch_num):
        model.train()
        train_loss = 0.0

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = {k: v.to(device) for k, v in features.items()}
            labels = targets['y'].to(device).float()

            optimizer.zero_grad()
            _, logits = model(features)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            loss = model.build_loss(pred=logits, labels=labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            logging.info("Epoch %s - Batch %s - Loss %s", epoch, batch_idx, loss.item())
            if args.train_batch_num & (batch_idx + 1) == args.train_batch_num:
                break

        if args.train_batch_num:
            re_train_nums = min(args.train_batch_num, len(train_loader))
        else:
            re_train_nums = len(train_loader)

        logging.info("Epoch %s - Loss %s - Total avg Loss %s", epoch, loss.item(), train_loss / re_train_nums)

        # evaluate
        eval_metrics = evaluate(model, eval_loader, device)
        eval_metrics_loss = eval_metrics['loss']
        logging.info("Eval Avg Loss %s", eval_metrics_loss)
        logging.info("Eval Avg AUC %s", eval_metrics['auc'])

        # save the best model
        if eval_metrics_loss < best_eval_loss:
            best_eval_loss = eval_metrics_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logger.info(f"No improvement for {no_improve_epochs} epochs. "
                        f"Best Eval Loss: {best_eval_loss:.4f}")
            if no_improve_epochs > args.early_stop_patience:
                logger.info(f"Early stop at epoch {epoch}")
                break


def evaluate(model: ETA, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(dataloader):
            # to device
            features = {k: v.to(device) for k, v in features.items()}
            labels = targets['y'].to(device).float()
            probs, logits = model(features)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            loss = model.build_loss(pred=logits, labels=labels)
            logging.info("Eval Batch Loss %s", loss.item())

            # collect the results
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

            if args.test_batch_num & (batch_idx + 1) == args.test_batch_num:
                break

    # calculate the indicators
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_logits = np.concatenate(all_logits)

    avg_loss = total_loss / len(all_labels)

    auc = roc_auc_score(all_labels, all_probs)

    return{
        'loss': avg_loss,
        'auc': auc,
    }


def collate_fn(batch):
    input_dicts = [item[0] for item in batch]
    target_dicts = [item[1] for item in batch]
    input_tensors = {}
    for key in input_dicts[0].keys():
        tensors = [d[key] for d in input_dicts if key in d and d[key] is not None]
        if not tensors:
            continue
        if tensors[0].dim() == 0:
            tensors = [t.unsqueeze(0) for t in tensors]
        input_tensors[key] = pad_sequence(tensors, batch_first=True)

    target_tensors = {}
    for key in target_dicts[0].keys():
        tensors = [d[key] for d in target_dicts if key in d and d[key] is not None]
        if not tensors:
            continue
        target_tensors[key] = torch.stack(tensors)

    return input_tensors, target_tensors


def main(args):
    args.model_dir = args.model_dir + datetime.now(china_tz).strftime('%Y%m%d')
    logger.info("Preparing for data loaders...")
    train_order = json_file_load("train_order", "./order.json")

    train_files = [
        "%strain/data_train.csv.hd5.%s" % (args.data_dir, index)
        for index in train_order["reading_order"]
    ]
    eval_files = glob.glob("%sval/data_val.csv.hd5.*" % args.data_dir)
    test_files = glob.glob("%stest/data_test.csv.hd5.*" % args.data_dir)

    if args.clear_existing_model and os.path.exists(args.model_dir):
        try:
            shutil.rmtree(args.model_dir)
        except PermissionError as e:
            raise PermissionError("Permission denied: {}".format(e)) from e
        except Exception as e:
            raise RuntimeError("Error clearing existing model: {}".format(e)) from e

    # ------ for NPU  ------
    train_dataset = TorchDataSet(train_files)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              prefetch_factor=100,
                              num_workers=10)
    eval_dataset = TorchDataSet(eval_files)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             collate_fn=collate_fn,
                             prefetch_factor=100,
                             num_workers=10)

    spec = json_file_load("spec", os.path.join(args.data_dir, "spec.json"))

    model = ETA(spec, args)
    device_type = 'npu'
    if not torch.npu.is_available() and torch.cuda.is_available():
        device_type = "cuda"
    elif not (torch.npu.is_available() or torch.cuda.is_available()):
        device_type = "cpu"
    device = torch.device(device_type)
    model.to(device)
    if args.task_type == "train":
        logger.info("start train and evaluate")
        train(model, train_loader, eval_loader, device, args)
        torch.save(model.load_state_dict, "eta.pth")
        logger.info(f"\n{'='*30}")
        logger.info("early stopped, start evaluating....")
        test_dataset = TorchDataSet(test_files[1:2])
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 prefetch_factor=100,
                                 num_workers=10)
        ret = evaluate(model, test_loader, device)
        logger.info("early stopped, end evaluating, result:%s", ret)
    else:
        raise ValueError("Unsupported task type: {}".format(args.task_type))


if __name__ == "__main__":
    args = parse_arguments()
    logger = logging.getLogger()
    log_level = getattr(logging, args.log_level.upper(), logging.DEBUG)
    logger.setLevel(log_level)
    console_hand = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
    console_hand.setLevel(log_level)
    console_hand.setFormatter(formatter)
    logger.addHandler(console_hand)
    # Define the timezone for China Standard Time
    china_tz = pytz.timezone('Asia/Shanghai')
    logfile_path = MODEL_NAME + "_" + datetime.now(china_tz).strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("FLAGS: " + str(args))
    feature_descriptions = {}
    logging.basicConfig(level=logging.INFO)
    main(args)

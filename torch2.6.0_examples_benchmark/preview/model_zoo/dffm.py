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
import torch_npu
from sklearn.metrics import roc_auc_score

# 该模型由examples/rec_model_zoo/behaviour_and_multi_task/src/models/dffm.py迁移而来，所有常量参数与其保持一致
torch.manual_seed(2024)
random.seed(2024)

MODEL_NAME = "DFFM"
TARGET_FIELDS = ["206", "207", "216", "210"]
SPARSE_FIELDS = ["101", "109_14", "110_14", "127_14", "150_14", "121", "122", "124", "125", "126", "127", "128", "129",
    "205", "206", "207", "210", "216", "508", "509", "702", "853"]
STD_DEV = (2 / 512) ** 0.5

@dataclass
class DFUBLayerEmbeddings:
    q_tar_embedding: torch.Tensor
    k_tar_embedding: torch.Tensor
    v_tar_embedding: torch.Tensor


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


class DFFMModel(nn.Module):
    def __init__(self, spec, params, device):
        super(DFFMModel, self).__init__()
        self.spec = spec
        self.params = params
        self.device = device

        self.emb_weights = nn.ModuleDict()
        self.target_dfub_weights = nn.ModuleDict()
        self.domain_dfub_weight = nn.Embedding(self.spec["vocab_length"]["301"] + 1,
            self.params.embedding_size * self.params.internal_size)
        nn.init.normal_(self.domain_dfub_weight.weight, std=STD_DEV)
        
        self.row = []
        self.col = []
        for i in range(22 - 1):
            for j in range(i + 1, 22):
                self.row.append(i)
                self.col.append(j)

        for key, vocab_len in self.spec["vocab_length"].items():
            self.emb_weights[key] = nn.Embedding(vocab_len + 1, self.params.embedding_size)
            nn.init.normal_(self.emb_weights[key].weight, std=STD_DEV)

        for target_key in TARGET_FIELDS:
            self.target_dfub_weights[target_key] = nn.Embedding(self.spec["vocab_length"][target_key] + 1,
                self.params.embedding_size * self.params.internal_size)
            nn.init.normal_(self.target_dfub_weights[target_key].weight, std=STD_DEV)

        self.meta_dnn_hidden_units = [self.params.embedding_size] + self.meta_dnn_hidden_units
        self.domain_map_mlp = nn.Sequential(
            nn.Linear(params.embedding_size, sum([self.meta_dnn_hidden_units[i] *
                self.meta_dnn_hidden_units[i + 1] for i in range(len(self.meta_dnn_hidden_units) - 1)])),
            nn.ReLU()
        )

        self.final_layer = nn.Linear(list(map(int, self.params.deep_layers.strip().split(',')))[-1], 1)

    def embedding_lookup_sparse_fake(self, embedding_layer,
                                     ids: torch.Tensor,
                                     combiner: str = None,
                                     name: str = None) -> torch.Tensor:
        dense_mask = torch.unsqueeze(torch.where(ids >= 0,
                                                 torch.ones_like(ids, dtype=torch.float32),
                                                 torch.zeros_like(ids)),
                                    dim=-1)

        ids = torch.where(ids == -1, torch.zeros_like(ids), ids)
        embedding_output = embedding_layer(ids)
        embedding = embedding_output * dense_mask
        summed_embedding = torch.sum(embedding, axis=1)
        if combiner == "sum":
            return summed_embedding
        elif combiner == "mean":
            return summed_embedding / torch.sum(dense_mask, axis=1)
        else:
            raise ValueError("combiner only support 'sum', 'mean'")

    
    def forward(self, features):
        embeddings = {key: self.emb_weights[key](features[key]) for key in self.spec["one_hot_fields"]}
        for key in self.spec["multi_hot_fields"] + self.spec["special_fields"]:
            embeddings[key] = self.embedding_lookup_sparse_fake(self.emb_weights[key], features[key], combiner="sum")

        sparse_input = torch.stack([embeddings[field_name] for field_name in SPARSE_FIELDS], dim=1)

        domain_emb = embeddings["301"]
        domain_emb = torch.relu(domain_emb)
        domain_vec = self.domain_map_mlp(domain_emb)

        weight_list = []
        bias_list = []
        offset = 0
        for i in range(len(self.meta_dnn_hidden_units) - 1):
            domain_weight = domain_vec[:, offset:offset + self.meta_dnn_hidden_units[i] *
                self.meta_dnn_hidden_units[i + 1]].view(
                    -1, self.meta_dnn_hidden_units[i], self.meta_dnn_hidden_units[i + 1])
            offset += self.meta_dnn_hidden_units[i] * self.meta_dnn_hidden_units[i + 1]
            weight_list.append(domain_weight)

        bias_list = [0.0] * len(weight_list)

        dffi_output = sparse_input
        for weight_i, weight in enumerate(weight_list):
            dffi_output = torch.einsum('ijk,ikl->ijl', dffi_output, weight) + bias_list[weight_i]
            if weight_i < len(weight_list) - 1:
                dffi_output = torch.relu(dffi_output)

        i_all_embeddings = torch.cat([dffi_output, sparse_input], dim=2)

        p = i_all_embeddings[:, self.row, :]
        q = i_all_embeddings[:, self.col, :]
        inner_product = torch.sum(p * q, dim=2)

        d_layer_input = torch.cat(
            [inner_product, i_all_embeddings.view(-1, 22 * self.params.embedding_size * 2)], dim=-1)

        dfub_target_emb = {}
        for target_key in TARGET_FIELDS:
            if target_key == "210":
                dfub_target_emb[target_key] = self.embedding_lookup_sparse_fake(self.target_dfub_weights[target_key],
                    features[target_key], combiner="sum").unsqueeze(1)
            else:
                dfub_target_emb[target_key] = self.target_dfub_weights[target_key](features[target_key]).view(
                    -1, 1, self.params.embedding_size * self.params.internal_size)

        dfub_his_emb = {}
        dfub_his_len = {}
        for his_key in self.spec["multi_hot_fields"]:
            feature_dense = features[his_key]
            dfub_his_len[his_key] = torch.sum((feature_dense >= 0).int(), dim=1, keepdim=True)
            dense_mask = (feature_dense >= 0).float().unsqueeze(-1)
            feature_dense = torch.where(feature_dense == -1, torch.zeros_like(feature_dense), feature_dense)
            emb = self.emb_weights[his_key](feature_dense)
            dfub_his_emb[his_key] = emb * dense_mask

        dfub_domain_emb = self.domain_dfub_weight(features["301"]).view(
            [-1, 1, self.params.embedding_size * self.params.internal_size])
        dfub_output = []
        for target_key, his_key in zip(TARGET_FIELDS, self.spec["multi_hot_fields"]):
            part_target_emb = dfub_target_emb[target_key]
            his_emb = dfub_his_emb[his_key]
            hist_mask = dfub_his_len[his_key]
            part1_num = int(0.5 * self.params.embedding_size * self.params.internal_size)
            part2_num = self.params.embedding_size * self.params.internal_size - part1_num
            target_emb = torch.cat([part_target_emb[:, :, :part1_num], dfub_domain_emb[:, :, :part2_num]], dim=-1)
            target_emb_c = DFUBLayerEmbeddings(q_tar_embedding=target_emb,
                                               k_tar_embedding=target_emb,
                                               v_tar_embedding=target_emb)

            his_emb_new = self.dfub_layer(his_emb, hist_mask, target_emb_c, scope=f"{target_key}_{his_key}")
            his_emb_new = torch.sum(his_emb_new, dim=1)
            dfub_output.append(his_emb_new)
        
        dfub_output_emb = torch.cat(dfub_output, dim=-1)

        d_layer_output = torch.cat([d_layer_input, dfub_output_emb], dim=-1)
        deep_layers = list(map(int, self.params.deep_layers.strip().split(',')))
        fullLayers = nn.ModuleList()
        num_input = d_layer_output.shape[-1]
        for layer_i, num_outputs in enumerate(deep_layers):
            fullLayers.append(nn.Linear(num_input, num_outputs).to(self.device))
            fullLayers.append(nn.ReLU().to(self.device))
            num_input = num_outputs

        for layer in fullLayers:
            d_layer_output = layer(d_layer_output)

        y = self.final_layer(d_layer_output).squeeze(-1)
        pred = torch.sigmoid(y)

        return pred, y

    def dfub_layer(self, seqs, masks, embeddings_t: DFUBLayerEmbeddings, scope="targ_hist"):
        q_tar_embedding = embeddings_t.q_tar_embedding
        k_tar_embedding = embeddings_t.k_tar_embedding
        v_tar_embedding = embeddings_t.v_tar_embedding

        w_q = nn.Parameter(torch.randn(self.params.embedding_size, self.params.internal_size, device=self.device) * 0.1)
        w_k = nn.Parameter(torch.randn(self.params.embedding_size, self.params.internal_size, device=self.device) * 0.1)
        w_v = nn.Parameter(torch.randn(self.params.embedding_size, self.params.internal_size, device=self.device) * 0.1)
        w_res = nn.Parameter(torch.randn(
            self.params.embedding_size, self.params.embedding_size, device=self.device) * 0.1)

        queries = torch.tensordot(seqs, w_q, dims=([-1], [0]))
        keys = torch.tensordot(seqs, w_k, dims=([-1], [0]))
        values = torch.tensordot(seqs, w_v, dims=([-1], [0]))

        q_tar_embedding = q_tar_embedding.view(-1, self.params.internal_size, self.params.embedding_size)
        queries = torch.matmul(queries, q_tar_embedding)
        k_tar_embedding = k_tar_embedding.view(-1, self.params.internal_size, self.params.embedding_size)
        keys = torch.matmul(keys, k_tar_embedding)
        v_tar_embedding = v_tar_embedding.view(-1, self.params.internal_size, self.params.embedding_size)
        values = torch.matmul(values, v_tar_embedding)

        q_ = torch.cat(torch.split(queries, self.params.embedding_size // self.params.heads_num, dim=2), dim=0)
        k_ = torch.cat(torch.split(keys, self.params.embedding_size // self.params.heads_num, dim=2), dim=0)
        v_ = torch.cat(torch.split(values, self.params.embedding_size // self.params.heads_num, dim=2), dim=0)

        outputs = torch.matmul(q_, k_.transpose(-2, -1))

        query_masks = masks
        key_masks = masks

        range_tensor = torch.arange(seqs.shape[1], device=masks.device).unsqueeze(0)
        query_masks = (range_tensor < query_masks.unsqueeze(-1)).to(dtype=torch.float32)
        key_masks = (range_tensor < key_masks.unsqueeze(-1)).to(dtype=torch.float32)
        query_masks = query_masks.squeeze(1)
        key_masks = key_masks.squeeze(1)

        key_masks = key_masks.repeat(self.params.heads_num, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.shape[1], 1)

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)

        outputs = torch.where(key_masks == 1, outputs, paddings)
        outputs = outputs - torch.max(outputs, dim=-1, keepdim=True)[0]
        softmax_outputs = torch.softmax(outputs, dim=-1)

        query_masks = query_masks.repeat(self.params.heads_num, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.shape[1])

        softmax_outputs = softmax_outputs * query_masks

        result = torch.matmul(softmax_outputs, v_)
        result = torch.cat(torch.split(result, result.shape[0] // self.params.heads_num, dim=0), dim=2)
        result += torch.tensordot(seqs, w_res, dims=([-1], [0]))
        result = torch.relu(result)
        return result


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


def train(model, train_loader, eval_loader, device, args):
    optimizer = model.build_optimizer()
    best_eval_loss = float('inf')
    no_improve_epochs = 0

    logger.info("Starting model trainning...")
    logger.info(f"Device: {device}")
    logger.info(f"Trainning config: {json.dumps(vars(args), indent=2)}")

    prof_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[
            torch_npu.profiler.ExportType.Text,
            torch_npu.profiler.ExportType.Db],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None)

    warmup_step_num = 1
    exec_step_num = 6
    prof_output_dir = os.path.join('./dffm.prof')
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup_step_num, active=exec_step_num, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_output_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=prof_config)

    prof.start()
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
            prof.step()
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
        eval_metrics = evaluate(model, eval_loader, device, prof)
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
    prof.stop()


def evaluate(model, dataloader, device, prof=None):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(dataloader):
            features = {k: v.to(device) for k, v in features.items()}
            labels = targets['y'].to(device).float()
            probs, logits = model(features)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            if prof is not None:
                prof.step()
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

    return { 'loss': avg_loss, 'auc': auc }


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
            input_tensors[key] = torch.cat(tensors, dim=0)
            continue
        input_tensors[key] = pad_sequence(tensors, batch_first=True)

    target_tensors = {}
    for key in target_dicts[0].keys():
        tensors = [d[key] for d in target_dicts if key in d and d[key] is not None]
        if not tensors:
            continue
        target_tensors[key] = torch.stack(tensors)

    return input_tensors, target_tensors


def parse_arguments():
    parser = argparse.ArgumentParser(description="DFFM Model Configuration")

    # Model parameters
    parser.add_argument("--embedding_size", type=int, default=16, help="Embedding size")
    parser.add_argument("--internal_size", type=int, default=8, help="Internal size")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer type {Adam, Adagrad, GD, Momentum}")
    parser.add_argument("--heads_num", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--deep_layers", type=str, default="512,256,128,64", help="Deep layers")

    # Data and model directories
    parser.add_argument("--dt_dir", type=str, default='', help="Data dt partition")
    parser.add_argument("--model_dir", type=str, default=f"./", help="Model check point dir")
    parser.add_argument("--servable_model_dir", type=str, default=f"./", help="Servable model directory")

    # Other configurations
    parser.add_argument("--clear_existing_model", action="store_true", help="Clear existing model or not")
    parser.add_argument("--log_level", type=str, default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Log level")
    parser.add_argument("--data_dir", type=str, default="/aliccp/aliccp_out", help="data dir")
    parser.add_argument("--task_type", type=str, default="train",
                        choices=["train", "eval", "predict"], help="task type")
    parser.add_argument('--epoch_num', type=int, default=1, help="Number of epochs")
    parser.add_argument('--train_batch_num', type=int, default=2000, help="Number of train batchs")
    parser.add_argument('--eval_batch_num', type=int, default=20, help="Number of eval batchs")
    parser.add_argument('--test_batch_num', type=int, default=20, help="Number of test batchs")
    return parser.parse_args()


def main(args):
    args.model_dir = args.model_dir + datetime.now(china_tz).strftime('%Y%m%d')
    logger.info("Preparing for data loaders...")
    train_order = json_file_load("train_order", "./order.json")

    train_files = glob.glob("%strain/data_train.csv.hd5.00000*" % args.data_dir)
    eval_files = glob.glob("%sval/data_val.csv.hd5.00000*" % args.data_dir)
    test_files = glob.glob("%stest/data_test.csv.hd5.00000*" % args.data_dir)

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

    device_type = 'npu'
    if not torch.npu.is_available() and torch.cuda.is_available():
        device_type = "cuda"
    elif not (torch.npu.is_available() or torch.cuda.is_available()):
        device_type = "cpu"
    model = DFFMModel(spec, args, device)
    model.to(device)
    if args.task_type == "train":
        logger.info("start train and evaluate")
        train(model, train_loader, eval_loader, device, args)
        torch.save(model.load_state_dict, "dffm.pth")
        logger.info(f"\n{'='*30}")
        logger.info("early stopped, start evaluating....")
        test_dataset = TorchDataSet(test_files)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 prefetch_factor=100,
                                 num_workers=10)
        evaluate(model, test_loader, device)
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
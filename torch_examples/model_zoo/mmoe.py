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

import os
import stat
import glob
import json
import random
import shutil
import logging
from datetime import datetime
import argparse

import pytz
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.metrics import roc_auc_score

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if torch.cuda.is_available():
    torch.cuda.manual_seed_all2024
else:
    torch.manual_seed(2024)
random.seed(2024)

MODEL_NAME = "MMOE"
PATIENCE = 5 # 早停参数，连续5轮训练loss都不减小，即停止训练。
EMBEDDING_FEATURE_NUM = 23 # sparse feature class num
STD_DEV = (2 / 512) ** 0.5 # 初始化权重标准差


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
        self.positive_indices = np.where(y == 1)[0]
        self.negative_indices = np.where(y == 0)[0]
        logger.info(f"load file {self.hdf5_path} finished")


class TorchDataSet(ConcatDataset):
    def __init__(self, files):
        datasets = [HDF5Dataset(fp) for fp in files]
        super().__init__(datasets)


def define_flags():
    parser = argparse.ArgumentParser(description='PyTorch Example with Command Line Arguments')
    parser.add_argument('--embedding_size', type=int, default=16, help="Embedding size")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "Adagrad", "GD", "Momentum"],
                        help="Optimizer type")
    parser.add_argument('--expert_layers', type=str, default="512,256", help="Expert layers")
    parser.add_argument('--tower_layers', type=str, default="128,64", help="tower layers")
    parser.add_argument('--ctr_task_wgt', type=float, default=0.5, help="loss weight of ctr task")
    parser.add_argument('--data_dir', type=str, default="alicpp/aliccp_out", help="Data directory")
    parser.add_argument('--dt_dir', type=str, default="", help="Data dt partition")
    parser.add_argument('--model_dir', type=str, default=f"./",
                        help="Model checkpoint directory")
    parser.add_argument('--servable_model_dir', type=str, default=f"./",
                        help="Export servable model for pytorch Serving")
    parser.add_argument('--task_type', type=str, default="train", choices=["train", "eval", "predict"],
                        help="Task type")
    parser.add_argument('--clear_existing_model', action="store_true", help="Clear existing model or not")
    parser.add_argument('--max_seq_len', type=int, default=50, help="Max length of sequence")
    parser.add_argument('--task_num', type=int, default=2, help="Task number")
    parser.add_argument('--experts_num', type=int, default=8, help="Number of experts")
    parser.add_argument('--log_level', type=str, default="DEBUG",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Log level")
    parser.add_argument('--epoch_num', type=int, default=10, help="Number of epochs")
    parser.add_argument('--train_batch_num', type=int, default=2000, help="Number of train batchs")
    parser.add_argument('--eval_batch_num', type=int, default=20, help="Number of eval batchs")
    parser.add_argument('--test_batch_num', type=int, default=20, help="Number of test batchs")
    return parser.parse_args()


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
    except Exception as e:
        raise RuntimeError(f"Error loading {json_name} file: {e}") from e

    return json_re


class TorchMmoeModel(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        # 获取数据结构配置，处理数据集自动生成
        self.spec = json_file_load("spec", os.path.join(self.params.data_dir, "spec.json"))
        self.embedding_layers = self.build_embedding_layers()
        self.experts = self.build_experts()
        self.gates = self.build_gate_networks()
        self.task_output_layers = self.build_task_output_layers()

        self.tower_names = ['ctr', 'cvr']
        self.towers = nn.ModuleDict()
        self.towers_output_layers = nn.ModuleDict()
        tower_units = list(map(int, self.params.tower_layers.strip().split(',')))
        input_dim = self.params.experts_num * list(map(int, self.params.expert_layers.strip().split(',')))[-1]
        for name in self.tower_names:
            tower_layers = []
            in_dim = input_dim
            for out_dim in tower_units:
                tower_layers.append(nn.Linear(in_dim, out_dim))
                tower_layers.append(nn.BatchNorm1d(out_dim))
                tower_layers.append(nn.ReLU())
                tower_layers.append(nn.Dropout(p=0.2))
                in_dim = out_dim
            self.towers[name] = nn.Sequential(*tower_layers)
            self.towers_output_layers[name] = nn.Linear(in_dim, 1)

    def forward(self, features: dict):
        # Build the embedding layer
        x_deep = self.get_embedding(features)
        experts_out = [expert(x_deep) for expert in self.experts]
        experts_out = torch.stack(experts_out, dim=1)

        gate_outs = [gate(x_deep) for gate in self.gates]

        task_outputs = []
        for gate_network in gate_outs:
            gate_network = gate_network.unsqueeze(-1)
            task_out = torch.multiply(experts_out, gate_network)
            task_out_shape = list(task_out.shape)
            task_outputs.append(torch.reshape(task_out, shape=[-1, task_out_shape[1] * task_out_shape[2]]))
        return task_outputs

    def build_embedding_layers(self):
        embeddings = nn.ModuleDict()
        for key, vocab_len in self.spec["vocab_length"].items():
            embeddings[key] = nn.Embedding(vocab_len + 1, self.params.embedding_size)
            nn.init.normal_(embeddings[key].weight, std=STD_DEV)
        return embeddings

    def build_experts(self):

        experts = nn.ModuleList()
        expert_units = list(map(int, self.params.expert_layers.strip().split(',')))
        input_dim = self.params.embedding_size * EMBEDDING_FEATURE_NUM

        for _ in range(self.params.experts_num):
            expert_layers = []
            in_features = input_dim

            for out_features in expert_units:
                expert_layers.append(nn.Linear(in_features, out_features))
                expert_layers.append(nn.BatchNorm1d(out_features))
                expert_layers.append(nn.ReLU())
                expert_layers.append(nn.Dropout(0.2))
                in_features = out_features
            experts.append(nn.Sequential(*expert_layers))
        return experts

    def build_gate_networks(self):
        gates = nn.ModuleList()
        input_dim = self.params.embedding_size * EMBEDDING_FEATURE_NUM

        for _ in range(self.params.task_num):
            gate = nn.Sequential(
                nn.Linear(input_dim, self.params.experts_num),
                nn.Softmax(dim=1)
            )
            gates.append(gate)
        return gates

    def build_task_output_layers(self):
        task_output_layers = nn.ModuleList()
        expert_units = list(map(int, self.params.expert_layers.strip().split(',')))
        input_dim = self.params.experts_num * expert_units[-1]
        tower_units = list(map(int, self.params.tower_layers.strip().split(',')))
        for _ in range(self.params.task_num):
            tower = []
            in_features = input_dim
            for out_features in tower_units:
                tower.append(nn.Linear(in_features, out_features))
                tower.append(nn.ReLU())
                in_features = out_features
            task_output_layers.append(nn.Sequential(*tower))
        return task_output_layers

    def embedding_lookup_sparse_fake(self, key,
                                     ids: torch.Tensor,
                                     combiner: str = None,
                                     name: str = None) -> torch.Tensor:
        dense_mask = torch.unsqueeze(torch.where(ids >= 0,
                                                 torch.ones_like(ids, dtype=torch.float32),
                                                 torch.zeros_like(ids)
                                                 ),
                                     dim=-1
                                     )

        # Replace invalid IDs (-1) with zeros
        ids = torch.where(ids == -1, torch.zeros_like(ids), ids)
        embedding_layer = self.embedding_layers[key]
        embedding_output = embedding_layer(ids)
        embedding = embedding_output * dense_mask
        summed_embedding = torch.sum(embedding, axis=1)
        if combiner == "sum":
            return summed_embedding
        elif combiner == "mean":
            return summed_embedding / torch.sum(dense_mask, axis=1)
        else:
            raise ValueError("combiner only supoort 'sum', 'mean'")

    def get_embedding(self, features: dict) -> torch.Tensor:
        """
        Build the embedding layer for the model.

        Args:
            features (dict): The input features.
        Returns:
            torch.Tensor: The concatenated and reshaped embedding tensor.
        """
        on_hot_field_lst = self.spec.get("one_hot_fields")
        other_field_lst = self.spec.get("multi_hot_fields") + self.spec.get("special_fields")

        embeddings = {}
        for key in on_hot_field_lst:
            embedding_layer = self.embedding_layers[key]
            embeddings[key] = embedding_layer(features[key])
            embeddings[key] = torch.reshape(embeddings[key], [-1, 1, self.params.embedding_size])
        for key in other_field_lst:
            embeddings[key] = torch.unsqueeze(
                self.embedding_lookup_sparse_fake(key=key, ids=features[key], combiner="sum"),
                dim=1
            )
        embedding = torch.concat(
            [embeddings.get(field_name) for field_name in self.spec.get("one_hot_fields")] +
            [embeddings.get(field_name) for field_name in self.spec.get("multi_hot_fields")] +
            [embeddings.get(field_name) for field_name in self.spec.get("special_fields")],
            dim=2,
        )

        return torch.reshape(embedding, [-1, EMBEDDING_FEATURE_NUM * self.params.embedding_size])

    def build_tower(self, tower_input: torch.Tensor, name: str) -> torch.Tensor:
        """
        Build the tower network for a specific task.

        Args:
            tower_input (torch.Tensor): The input tensor for the tower.
            name (str): The name of the tower.
        Returns:
            torch.Tensor: The output tensor of the tower network.
        """
        tower_units = list(map(int, self.params.tower_layers.strip().split(',')))
        y_tower = tower_input
        for tower_i, _ in enumerate(tower_units):
            tower_linear = nn.Linear(in_features=y_tower.shape[-1], out_features=tower_units[tower_i])
            tower_linear_out = tower_linear(y_tower)
            relu = nn.ReLU()
            y_tower = relu(tower_linear_out)
        return y_tower

    def build_predictions(self, task_outputs: list) -> dict:
        """
        Build the predictions for the model.

        Args:
            task_outputs (list): A list of task output tensors.

        Returns:
            dict: A dictionary containing the predictions for ctr, cvr, and ctcvr.
        """
        preds = {}
        tower_outputs = {}
        for i, name in enumerate(self.tower_names):
            y = self.towers[name](task_outputs[i])
            y = self.towers_output_layers[name](y)
            y = torch.reshape(y, [-1, ])
            preds[name + '_logit'] = y
            preds[name] = torch.sigmoid(y)
            tower_outputs[name] = y

        ctr_pred = preds.get('ctr')
        cvr_pred = preds.get('cvr')
        if ctr_pred is not None and cvr_pred is not None:
            if 'ctr_logit' not in preds or 'cvr_logit' not in preds:
                raise ValueError("Missing required keys in preds dictionary")
            else:
                preds['ctcvr'] = ctr_pred * cvr_pred
                preds['ctcvr_logit'] = preds['ctr_logit'] + preds['cvr_logit']
        return preds

    def build_loss(self,
                   labels: dict,
                   y_ctr_logit: torch.Tensor,
                   y_ctcvr_logit: torch.Tensor,
                   ctr_weight,
                   ctcvr_weight) -> torch.Tensor:
        """
        Build the loss function for the model.

        Args:
            labels (dict): A dictionary containing the true labels for ctr and ctcvr.
            y_ctr_logit (torch.Tensor): The predicted ctr values.
            y_ctcvr_logit (torch.Tensor): The predicted ctcvr values.
        Returns:
            torch.Tensor: The combined loss tensor.
        """
        y_ctr_logit = y_ctr_logit.view(-1)
        y_ctcvr_logit = y_ctcvr_logit.view(-1)
        y = labels['y'].view(-1)
        z = labels['z'].view(-1)

        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ctr_weight],
                                                                dtype=torch.float32,
                                                                device=y.device)
                                        )
        ctcvr_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ctcvr_weight],
                                                                      dtype=torch.float32,
                                                                      device=z.device)
                                              )

        ctr_loss = bce_loss(y_ctr_logit, y)
        ctcvr_loss = ctcvr_bce_loss(y_ctcvr_logit, z)
        ctr_task_wgt = self.params.ctr_task_wgt

        return ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss

    def build_optimizer(self):
        """
        Build the optimizer for training.

        Args:
            loss (torch.Tensor): The loss tensor to minimize.

        Returns:
           optim: The operation for applying gradients.

        Raises:
            ValueError: If the optimizer type is not supported.
        """
        if self.params.optimizer == "Adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.params.learning_rate,
                betas=[0.9, 0.99], eps=1e-8
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


def clip_grad(grad):
    if grad is None:
        return grad
    return torch.clamp(grad, -1, 1)


def train(model: TorchMmoeModel, tr_files, va_files, args, device):
    optimizer = model.build_optimizer()
    epochs = args.epoch_num

    train_dataset = TorchDataSet(tr_files)
    ctr_percent_train, ctcvr_percent_train = pre_deal_dataset(train_dataset)
    logging.info("Train data ctr: %.4f, ctcvr: %.4f", ctr_percent_train, ctcvr_percent_train)
    dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  prefetch_factor=100,
                                  num_workers=10)
    va_dataset = TorchDataSet(va_files)
    val_ctr, val_ctcvr = pre_deal_dataset(va_dataset)
    logging.info("Eval data ctr: %.4f, ctcvr: %.4f", val_ctr, val_ctcvr)
    val_dataloader = DataLoader(dataset=va_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               collate_fn=collate_fn,
                               prefetch_factor=100,
                               num_workers=10)

    # 早停相关变量
    best_auc = float("inf")
    counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        now_index = 0
        for input_sample, target_sample in dataloader:
            input_sample = {k: v.to(device) for k, v in input_sample.items()}
            target_sample = {k: v.to(device) for k, v in target_sample.items()}
            optimizer.zero_grad()
            task_outputs = model.forward(input_sample)
            predictions = model.build_predictions(task_outputs)
            loss = model.build_loss(target_sample,
                                    predictions["ctr_logit"],
                                    predictions["ctcvr_logit"],
                                    ctr_percent_train,
                                    ctcvr_percent_train)
            loss.backward()
            nn.utils.clip_grad.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            logging.info("Epoch %s - Batch %s - Loss %s", epoch, now_index, loss.item())

            now_index += 1
            if args.train_batch_num and now_index == args.train_batch_num:
                break
        if args.train_batch_num:
            re_train_nums = min(args.train_batch_num, len(dataloader))
        else:
            re_train_nums = len(dataloader)
        logging.info("Epoch %s - Loss %s - Total avg Loss %s", epoch, loss.item(), total_loss / re_train_nums)

        model.eval()
        val_loss = 0.0
        val_index = 0
        with torch.no_grad():
            for eval_input_sample, eval_target_sample in val_dataloader:
                eval_input_sample = {k: v.to(device) for k, v in eval_input_sample.items()}
                eval_target_sample = {k: v.to(device) for k, v in eval_target_sample.items()}

                task_outputs = model.forward(eval_input_sample)
                predictions = model.build_predictions(task_outputs)
                loss = model.build_loss(eval_target_sample,
                                        predictions["ctr"],
                                        predictions["ctcvr"],
                                        val_ctr,
                                        val_ctcvr)
                val_loss += loss.item()
                y_true = eval_target_sample["y"].cpu().numpy()
                z_true = eval_target_sample["z"].cpu().numpy()
                ctr_auc = roc_auc_score(y_true, predictions["ctr"].detach().cpu().numpy())
                ctcvr_auc = roc_auc_score(z_true, predictions["ctcvr"].detach().cpu().numpy())
                logging.info("Eval Batch Loss %s - Ctr Auc %s - Ctcvr Auc %s", loss.item(), ctr_auc, ctcvr_auc)

                val_index += 1
                if args.eval_batch_num and val_index == args.eval_batch_num:
                    break

        if args.eval_batch_num:
            re_eval_nums = min(args.eval_batch_num, len(val_dataloader))
        else:
            re_eval_nums = len(val_dataloader)

        avg_val_loss = val_loss / re_eval_nums
        logging.info("Eval Avg Loss %s", avg_val_loss)

        if best_auc < ctcvr_auc:
            best_auc = ctcvr_auc
            counter = 0
        else:
            counter += 1
            if counter > PATIENCE:
                logging.info("Early stop at epoch %s", epoch)
                break


def evaluate(model: TorchMmoeModel, te_files, device):
    model.eval()
    total_loss = 0.0
    all_ctr_labels = []
    all_ctr_preds = []
    all_cvr_labels = []
    all_cvr_preds = []
    all_ctcvr_labels = []
    all_ctcvr_preds = []

    te_dataset = TorchDataSet(te_files)
    te_ctr, te_ctcvr = pre_deal_dataset(te_dataset)
    logging.info("Test data ctr: %.4f, ctcvr: %.4f", te_ctr, te_ctcvr)
    test_dataloader = DataLoader(dataset=te_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               collate_fn=collate_fn,
                               prefetch_factor=100,
                               num_workers=10)

    test_index = 0
    with torch.no_grad():
        for input_sample, target_sample in test_dataloader:
            input_sample = {k: v.to(device) for k, v in input_sample.items()}
            target_sample = {k: v.to(device) for k, v in target_sample.items()}

            task_outputs = model.forward(input_sample)
            predictions = model.build_predictions(task_outputs)
            loss = model.build_loss(target_sample,
                                    predictions["ctr_logit"],
                                    predictions["ctcvr_logit"],
                                    te_ctr,
                                    te_ctcvr)
            total_loss += loss.item()
            # CTR
            all_ctr_labels.append(target_sample["y"].detach().cpu().numpy())
            all_ctr_preds.append(predictions["ctr"].detach().cpu().numpy())
            # CTCVR
            all_ctcvr_labels.append(target_sample["z"].detach().cpu().numpy())
            all_ctcvr_preds.append(predictions["ctcvr"].detach().cpu().numpy())
            # CVR 只保留点击的
            mask = (target_sample["y"].squeeze(-1) > 0)
            if mask.sum() > 0:
                all_cvr_labels.append(target_sample["z"][mask].detach().cpu().numpy())
                all_cvr_preds.append(predictions["cvr"][mask].detach().cpu().numpy())

            logging.info("Test Batch Loss %s", loss.item())
            test_index += 1

            if args.test_batch_num and test_index == args.test_batch_num:
                break

    if args.test_batch_num:
        re_test_nums = min(args.test_batch_num, len(test_dataloader))
    else:
        re_test_nums = len(test_dataloader)

    avg_test_loss = total_loss / re_test_nums
    logging.info("Test Avg Loss: %.4f", avg_test_loss)


    all_ctr_labels = np.concatenate(all_ctr_labels)
    all_ctr_preds = np.concatenate(all_ctr_preds)
    all_ctcvr_labels = np.concatenate(all_ctcvr_labels)
    all_ctcvr_preds = np.concatenate(all_ctcvr_preds)

    if all_cvr_labels:
        all_cvr_labels = np.concatenate(all_cvr_labels)
        all_cvr_preds = np.concatenate(all_cvr_preds)

    auc_ctr = roc_auc_score(all_ctr_labels, all_ctr_preds)
    auc_ctcvr = roc_auc_score(all_ctcvr_labels, all_ctcvr_preds)
    auc_cvr = roc_auc_score(all_cvr_labels, all_cvr_preds) if len(all_cvr_labels) > 0 else float("nan")

    logging.info("AUC CTR: %.4f", auc_ctr)
    logging.info("AUC CVR: %.4f", auc_cvr)
    logging.info("AUC CTCVR: %.4f", auc_ctcvr)

    return avg_test_loss


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


def pre_deal_dataset(dataset):
    num_pos = 0
    num_neg = 0
    num_pos_y = 0
    num_neg_y = 0
    dataset_index = 0
    for sub_ds in dataset.datasets:
        z = sub_ds.target_sample["z"]
        num_pos += (z == 1).sum().item()
        num_neg += (z == 0).sum().item()

        y = sub_ds.target_sample["y"]
        num_pos_y += (y == 1).sum().item()
        num_neg_y += (y == 0).sum().item()

        logger.info("find pos index %s", dataset_index)
        dataset_index += 1
    return np.log(num_neg_y / num_pos_y), np.log(num_neg / num_pos)



def main(args):
    args.model_dir = args.model_dir + datetime.now(china_tz).strftime('%Y%m%d')

    train_order = json_file_load("train_order", "./order.json")

    tr_files = [
        "%strain/data_train.csv.hd5.%s" % (args.data_dir, index)
        for index in train_order["reading_order"]
    ]
    va_files = glob.glob("%sval/data_val.csv.hd5.*" % args.data_dir)
    te_files = glob.glob("%stest/data_test.csv.hd5.*" % args.data_dir)

    if args.clear_existing_model and os.path.exists(args.model_dir):
        try:
            shutil.rmtree(args.model_dir)
        except PermissionError as e:
            raise PermissionError("Permission denied: {}".format(e)) from e
        except Exception as e:
            raise RuntimeError("Error clearing existing model: {}".format(e)) from e

    # ------ for NPU  ------

    model = TorchMmoeModel(args)
    device_type = 'npu'
    if not torch.npu.is_available() and torch.cuda.is_available():
        device_type = "cuda"
    elif not (torch.npu.is_available() or torch.cuda.is_available()):
        device_type = "cpu"
    device = torch.device(device_type)
    model.to(device)
    if args.task_type == "train":
        logger.info("start train and evaluate")
        train(model, tr_files, va_files, args, device)
        torch.save(model.load_state_dict, "mmoe.pth")
        logger.info("early stopped, start evaluating....")

        evaluate(model, te_files, device)
    else:
        raise ValueError("Unsupported task type: {}".format(args.task_type))


if __name__ == "__main__":

    args = define_flags()
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

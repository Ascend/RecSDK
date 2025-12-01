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
import json
import stat
import glob
import time

import tensorflow as tf
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from utils.logger import logger
from utils.common import Profiler, save_json
from utils.handler import TestHandler


class AliccpDataset(Dataset):
    def __init__(
        self, params, filepath, feature_description, spec, mode, shuffle=False
    ):
        self.params = params
        self.spec = spec
        self.filepath = filepath
        self.feature_description = feature_description
        self.dataset = tf.data.TFRecordDataset(self.filepath)
        if shuffle:
            self.dataset.shuffle(buffer_size=500000)
        self.dataset = (
            self.dataset.repeat(params.num_epochs)
            .batch(self.params.batch_size, drop_remainder=True)
            .map(self.parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(100)
        )
        self.iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        self.length = spec["dataset_size"][mode] // self.params.batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_features, batch_labels = self.iterator.get_next()
        for key, val in batch_features.items():
            batch_features[key] = torch.tensor(val.numpy(), dtype=torch.long).to(
                self.params.device
            )
        for key, val in batch_labels.items():
            batch_labels[key] = torch.tensor(val.numpy(), dtype=torch.float32).to(
                self.params.device
            )
        return batch_features, batch_labels

    def parse_example(self, example):
        parsed_example = tf.io.parse_example(example, self.feature_description)
        input_data = {}
        target = {"y": parsed_example["y"], "z": parsed_example["z"]}
        for index, key in enumerate(self.spec["one_hot_fields"]):
            input_data[key] = parsed_example["one_hot_fields"][:, index]
        for key in self.spec["multi_hot_fields"]:
            input_data[key] = parsed_example[key]
        for key in self.spec["special_fields"]:
            input_data[key] = parsed_example[key]
        return input_data, target


def cal_auc(pred, labels):
    return roc_auc_score(labels, pred)


def get_spec(params):
    root_path = os.path.abspath(__file__)
    root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
    spec_json_path = os.path.join(root_path, "datasets/aliccp", "spec.json")
    local_spec = json_file_load("spec", spec_json_path)
    logger.info(f"spec_json_path: {spec_json_path}")
    if params.mode == "test_qps":
        one_hot_list = []
        multi_hot_list = []
        one_hot_num = len(local_spec["one_hot_fields"])
        multi_hot_num = len(local_spec["multi_hot_fields"])
        one_hots = int(
            params.extra_fields * (one_hot_num / (one_hot_num + multi_hot_num))
        )
        multi_hots = params.extra_fields - one_hots
        params.extra_multi_hots = multi_hots
        logger.info(one_hots)

        # 先同步生成one_hot和multi_hot特征，多余的生成one_hot特征
        for i in range(multi_hots):
            one_hot_list.append(f"{1000+i}")
            multi_hot_list.append(f"{1000+i}_14")
        for i in range(multi_hots, params.extra_fields):
            one_hot_list.append(f"{1000+i}")
        local_spec["multi_hot_fields"].extend(multi_hot_list)
        local_spec["one_hot_fields"].extend(one_hot_list)
        for key in multi_hot_list:
            local_spec["vocab_length"][key] = 10000
            local_spec["train_max_length"][key] = 50
            local_spec["test_max_length"][key] = 50
            local_spec["val_max_length"][key] = 50
        for key in one_hot_list:
            local_spec["vocab_length"][key] = 50
            local_spec["train_max_length"][key] = 1
            local_spec["test_max_length"][key] = 1
            local_spec["val_max_length"][key] = 1
    return local_spec


def build_feature_descriptions(local_spec):
    local_feature_descriptions = {}
    mode = ["train", "val", "test"]
    for mode_type in mode:
        feature_description = {
            "y": tf.io.FixedLenFeature([], tf.float32),
            "z": tf.io.FixedLenFeature([], tf.float32),
            "one_hot_fields": tf.io.FixedLenFeature(
                [len(local_spec["one_hot_fields"])], tf.int64
            ),
        }
        for mul_fields in local_spec["multi_hot_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature(
                [local_spec.get(f"{mode_type}_max_length").get(mul_fields)], tf.int64
            )
        for mul_fields in local_spec["special_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature(
                [local_spec.get(f"{mode_type}_max_length").get(mul_fields)], tf.int64
            )
        local_feature_descriptions[mode_type] = feature_description

    return local_feature_descriptions


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


def generate_dataloader(dataset):
    return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])


def load_data(params):
    spec = get_spec(params)
    feature_desc = build_feature_descriptions(spec)

    root_path = os.path.abspath(__file__)
    root_path = os.path.sep.join(root_path.split(os.path.sep)[:-1])
    train_order = json_file_load(
        "order", os.path.join(root_path, "aliccp", "order.json")
    )
    tr_files = []
    for index in train_order["reading_order"]:
        tr_files.append(
            os.path.join(
                params.data_dir, "train", "data_train.csv.tfrecord.{}".format(index)
            )
        )
    va_files = sorted(glob.glob(
        os.path.join(params.data_dir, "val", "data_val.csv.tfrecord.*")
    ))
    te_files = sorted(glob.glob(
        os.path.join(params.data_dir, "test", "data_test.csv.tfrecord.*")
    ))

    train_dataset = AliccpDataset(
        params,
        tr_files,
        feature_description=feature_desc.get("train"),
        spec=spec,
        mode="train",
        shuffle=True,
    )
    test_dataset = AliccpDataset(
        params,
        te_files,
        feature_description=feature_desc.get("test"),
        spec=spec,
        mode="test",
    )
    val_dataset = AliccpDataset(
        params,
        va_files,
        feature_description=feature_desc.get("val"),
        spec=spec,
        mode="val",
    )

    # batch_sizeand num_worker由内部处理
    train_loader = generate_dataloader(train_dataset)
    test_loader = generate_dataloader(test_dataset)
    val_loader = generate_dataloader(val_dataset)

    return train_loader, test_loader, val_loader


def load_generate_data(params):
    spec_json_path = os.path.join(params.data_dir, "spec.json")
    local_spec = json_file_load("spec", spec_json_path)
    return local_spec


class TestAliccpHandler(TestHandler):
    def __init__(self, params, spec):
        super().__init__(params)
        self.spec = spec

    def generate_data(self, batch_size):

        features = {}
        device = self.params.device
        for key in self.spec["one_hot_fields"]:
            features[key] = torch.randint(
                low=0, high=self.spec["vocab_length"][key], size=(batch_size, 1)
            )[:, 0].to(device)

        for key in self.spec["multi_hot_fields"]:
            features[key] = torch.randint(
                low=0, high=self.spec["vocab_length"][key], size=(batch_size, 50)
            ).to(device)

        for key in self.spec["special_fields"]:
            features[key] = torch.randint(
                low=0, high=self.spec["vocab_length"][key], size=(batch_size, 38)
            ).to(device)
        return features
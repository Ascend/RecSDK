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
import random
import json
import stat
import glob

import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

from utils.logger import logger
from utils.common import get_loop_element
from utils.handler import TestHandler
from sklearn.metrics import roc_auc_score

CAT_FEATURE_COUNT = 26
NUM_EMBEDDINGS_PER_FEATURE = [
    1000000,39060,17295,7424,20265,
    3,7122,1543,63,1000000,
    3067956,405282,10,2209,11938,
    155,4,976,14,1000000,
    1000000,1000000,590152,12973,108,36
]
MULTI_HOT_SIZES = [
    3,2,1,2,6,
    1,1,1,1,7,
    3,8,1,6,9,
    5,1,1,1,12,
    100,27,10,3,1,1
]

CRITEO_NUM = {"train": 33003326, "val": 8250124, "test": 4587176}

class CriteoDataset(Dataset):
    def __init__(self, params, filepath, mode, shuffle=False):
        self.params = params
        self.filepath = filepath
        self.feature_description = {
            "label": tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
            "ids": tf.io.FixedLenFeature(shape=(self.params.field_size,), dtype=tf.int64),
            "values": tf.io.FixedLenFeature(shape=(self.params.field_size,), dtype=tf.float32),
        }
        self.dataset = tf.data.TFRecordDataset(self.filepath)
        if shuffle:
            self.dataset.shuffle(buffer_size=5000)
        self.dataset = (
            self.dataset.repeat(params.num_epochs)
            .batch(self.params.batch_size, drop_remainder=True)
            .map(self.parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(50)
        )
        self.iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        self.length = CRITEO_NUM[mode] // self.params.batch_size

    def __len__(self):         
        return self.length

    def __getitem__(self, idx):
        batch_features, batch_labels = self.iterator.get_next()
        batch_features["feat_ids"] = torch.tensor(batch_features["feat_ids"].numpy(), dtype=torch.int32).to(
            self.params.device
        )
        batch_features["feat_vals"] = torch.tensor(batch_features["feat_vals"].numpy(), dtype=torch.float32).to(
            self.params.device
        )
        batch_labels = torch.tensor(batch_labels.numpy(), dtype=torch.float32).to(self.params.device)
        return batch_features, batch_labels
    
    def parse_example(self, example):         
        sample = tf.io.parse_example(example, self.feature_description)         
        sample["ids"] = tf.cast(sample["ids"], dtype=tf.int32)         
        return {"feat_ids": sample["ids"], "feat_vals": sample["values"]}, sample["label"]


def cal_auc(pred, labels):
    return roc_auc_score(labels, pred)


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
    tr_files = sorted(glob.glob(os.path.join(params.data_dir, "tr*tfrecords")))
    random.shuffle(tr_files)
    va_files = sorted(glob.glob(os.path.join(params.data_dir, "va*tfrecords")))
    te_files = sorted(glob.glob(os.path.join(params.data_dir, "te*tfrecords")))

    train_dataset = CriteoDataset(params, tr_files, mode="train", shuffle=True)
    test_dataset = CriteoDataset(params, te_files, mode="test")
    val_dataset = CriteoDataset(params, va_files, mode="val")

    # batch_sizeand num_worker由内部处理
    train_loader = generate_dataloader(train_dataset)
    test_loader = generate_dataloader(test_dataset)
    val_loader = generate_dataloader(val_dataset)
    return train_loader, test_loader, val_loader

class TestCriteoMultihotHandler(TestHandler):
    def __init__(self, params):
        super().__init__(params)
        
    
    def generate_data(self, batch_size):
        features = {}
        device = self.params.device
        feat_ids = []
        for i in range(self.params.multi_fields_count):             
            dim = get_loop_element(NUM_EMBEDDINGS_PER_FEATURE, i)            
            length = get_loop_element(MULTI_HOT_SIZES, i)             
            feat_ids.append(torch.randint(0, dim ,(batch_size, length)))
        features["feat_ids"] = torch.concat(feat_ids,dim=-1).to(device)
        features["feat_vals"] = torch.rand((batch_size, 13)).to(device)
        return features



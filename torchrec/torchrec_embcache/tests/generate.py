#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os

import numpy as np


np.random.seed(2)

KEYS_NUMS = [40, 20]
EMBED_DIM = 128
KEYS_TYPE_BYTES = 8
EMBED_TYPE_BYTEST = 4

TABLE_NAMES = ["table0", "table1"]
RANKS = [0, 1]
OUT_DIR = "sparse"


os.makedirs(OUT_DIR, exist_ok=True)


def do_out(table_name, rank_id, slice_name, attribute, data):
    path = os.path.join(OUT_DIR, table_name, f"rank{rank_id}", slice_name)
    os.makedirs(path, exist_ok=False)
    attribute_path = os.path.join(OUT_DIR, table_name, f"rank{rank_id}", slice_name, "slice.attribute")
    data_path = os.path.join(OUT_DIR, table_name, f"rank{rank_id}", slice_name, "slice.data")
    attribute.tofile(attribute_path)
    data.tofile(data_path)


keys_attributes = [np.array([KEYS_TYPE_BYTES, KEYS_NUM]).astype(np.int64) for KEYS_NUM in KEYS_NUMS]
embed_attributes = [np.array([EMBED_TYPE_BYTEST, KEYS_NUM, EMBED_DIM]).astype(np.int64) for KEYS_NUM in KEYS_NUMS]
moment1_attributes = [np.array([EMBED_TYPE_BYTEST, KEYS_NUM, EMBED_DIM]).astype(np.int64) for KEYS_NUM in KEYS_NUMS]

keys_datas = [np.arange(KEYS_NUM).astype(np.int64) for KEYS_NUM in KEYS_NUMS]
embed_datas = [
    np.broadcast_to(
            np.arange(KEYS_NUM)[:, np.newaxis], 
            (KEYS_NUM, EMBED_DIM)
        ).astype(np.float32) for KEYS_NUM in KEYS_NUMS
    ]
moment1_datas = [np.ones((KEYS_NUM, EMBED_DIM), dtype=np.float32) for KEYS_NUM in KEYS_NUMS]


for i, table in enumerate(TABLE_NAMES):
    keys_attribute = keys_attributes[i]
    keys_data = keys_datas[i]
    embed_attribute = embed_attributes[i]
    embed_data = embed_datas[i]
    moment1_attribute = moment1_attributes[i]
    moment1_data = moment1_datas[i]
    for rank in RANKS:
        if rank == 1 and i == 1:
            keys_attribute = np.array([KEYS_TYPE_BYTES, 0]).astype(np.int64)
            keys_data = np.arange(0).astype(np.int64)
            embed_attribute = np.array([EMBED_TYPE_BYTEST, 0, EMBED_DIM]).astype(np.int64)
            embed_data = np.ones((0, EMBED_DIM), dtype=np.float32)
            moment1_attribute = np.array([EMBED_TYPE_BYTEST, 0, EMBED_DIM]).astype(np.int64)
            moment1_data = np.ones((0, EMBED_DIM), dtype=np.float32)

        do_out(table, rank, "key", keys_attribute, keys_data)
        do_out(table, rank, "embedding", embed_attribute, embed_data)
        do_out(table, rank, "momentum1", moment1_attribute, moment1_data)

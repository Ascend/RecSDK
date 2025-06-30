#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import random
from copy import deepcopy

import pytest
import torch
import torch.multiprocessing as mp
from dataset import RandomRecDataset, Batch, BoundOutOfRangeRecDataset, FeatureNameNotInConfigRecDataset
from model import TestModel, generate_hash_config, HashConfig
from torch.optim import Adam, Adagrad
from torch.utils.data import DataLoader
from util import (
    is_lookup_out_of_bound,
    feature_name_exists,
    setup_logging,
    create_weight_init,
    check_config,
    OVER_COUNT,
    run_model_with_config
)

import torchrec


@pytest.mark.functional
def test_normal(config):
    run_model_with_config(config)


def execute(rank, config):
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    check_config(config)
    embedding_dims = config["embedding_dims"]
    num_embeddings = config["num_embeddings"]
    pool_type = config["pool_type"]
    batch_num = config["BATCH_NUM"]
    table_num = config["table_num"]
    lookup_lens = config["lookup_lens"]
    dataset_class = globals()[config["RecDataset"] + "RecDataset"]
    init_fn = globals()[config["init_fn"]]
    world_size = config["WORLD_SIZE"]
    device = config.get("device", "npu")
    sharding_type = config.get("sharding_type", "row_wise")
    optim = globals()[config.get("optim", "Adagrad")]
    feature_names_lst = config["feature_names_lst"]
    instances = config.get("instances", 1)
    pool_type = getattr(torchrec.PoolingType, pool_type)
    hash_config = HashConfig(
        embedding_dims=embedding_dims, 
        num_embeddings=num_embeddings, 
        pooling=pool_type, 
        feature_names=feature_names_lst,
        init_fn=create_weight_init(init_fn),
        collection_type="ebc"
    )
    embedding_config = generate_hash_config(hash_config)
    generated_ids = []
    if isinstance(dataset_class, BoundOutOfRangeRecDataset):
        for i in range(table_num):
            generated_ids.append([])
            for _ in range(len(feature_names_lst[i])):
                generated_ids[i].append(list(range(num_embeddings[i] + OVER_COUNT)))
                random.shuffle(generated_ids[i][-1])
    dataset_gloden = dataset_class(batch_num, lookup_lens, num_embeddings, table_num, feature_names_lst, generated_ids)
    dataset = dataset_class(
        batch_num, lookup_lens, num_embeddings, table_num, feature_names_lst, deepcopy(generated_ids)
    )
    data_loader_gloden = DataLoader(
        dataset_gloden,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    test_model = TestModel(rank, world_size, device, instances, feature_names_lst, batch_num, collection_type="ebc")

    golden_results = test_model.cpu_golden_loss(embedding_config, data_loader_gloden, optim)
    test_model.init_ddp_model(embedding_config, sharding_type, optim, lookup_lens)
    test_results = test_model.test_pipe_loss(data_loader)
    for golden, result in zip(golden_results, test_results):
        logging.debug("")
        logging.debug("===========================")
        logging.debug("result test %s", result)
        logging.debug("golden test %s", golden)
        assert torch.allclose(
            golden, result, rtol=1e-04, atol=1e-04
        ), "golden and result is not closed"
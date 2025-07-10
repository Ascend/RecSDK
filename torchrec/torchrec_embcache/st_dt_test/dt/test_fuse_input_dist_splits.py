#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
import os
from concurrent.futures import ProcessPoolExecutor

import pytest
import torch
import numpy as np
import torch.multiprocessing as mp
from dataset import (
    RandomRecDataset, 
    Batch, 
    BoundOutOfRangeRecDataset, 
    FeatureNameNotInConfigRecDataset
)
from dt.conftest import MODULE_NAME
from model import TestModel, generate_hash_config, HashConfig
from torch.utils.data import DataLoader
from torch.optim import Adam, Adagrad
from torchrec_embcache.distributed.train_pipeline import AwaitableAdapter, EmbCacheTrainPipelineContext
from util import (
    setup_logging,
    is_lookup_out_of_bound,
    feature_name_exists,
    create_weight_init,
    check_config,
    TEST_ROOT_DIR,
    OVER_COUNT,
    fuse_input_dist_splits,
    are_features_equal,
    run_model_with_config,
    DATASET_REGISTRY,
    INIT_FN_REGISTRY,
    OPTIM_REGISTRY
)

import torchrec
from torchrec import EmbeddingBagConfig, EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@pytest.mark.functional
def test_normal(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    run_model_with_config(config, execute)


@pytest.mark.functional
def test_table_num_invalid(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert config["table_num"] < 1
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config, execute)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "ValueError" in str(exc_info.value)


@pytest.mark.functional
def test_embedding_dim_invalid(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert any([embedding_dim < 1 or embedding_dim % 4 for embedding_dim in config["embedding_dims"]])
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config, execute)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "ValueError" in str(exc_info.value)


@pytest.mark.functional
def test_num_embeddings_invalid(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert any([num_embedding < 1 for num_embedding in config["num_embeddings"]])
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config, execute)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "ValueError" in str(exc_info.value)


# 只有多级缓存需要
@pytest.mark.functional
def test_lookup_out_of_bound(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert is_lookup_out_of_bound(config)
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config, execute)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "IndexError" in str(exc_info.value)


@pytest.mark.functional
def test_feature_name_exist(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert feature_name_exists(config)
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config, execute)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "KeyError" in str(exc_info.value)


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
    dataset_class = DATASET_REGISTRY.get(config["RecDataset"] + "RecDataset", RandomRecDataset)
    init_fn = INIT_FN_REGISTRY.get(config["init_fn"], create_weight_init("init_linspace"))
    world_size = config["WORLD_SIZE"]
    device = config.get("device", "npu")
    sharding_type = config.get("sharding_type", "row_wise")
    optim = OPTIM_REGISTRY.get(config.get("optim", "Adagrad"), Adagrad)
    feature_names_list = config["feature_names_list"]
    instances = config.get("instances", 1)
    pool_type = getattr(torchrec.PoolingType, pool_type)
    collection_type = config["collection_type"]
    hash_config = HashConfig(
        embedding_dims=embedding_dims,
        num_embeddings=num_embeddings,
        pool_type=pool_type,
        feature_names_list=feature_names_list,
        init_fn=create_weight_init(init_fn),
        collection_type=collection_type,
    )
    embedding_config = generate_hash_config(hash_config)
    generated_ids = []
    if dataset_class is BoundOutOfRangeRecDataset:
        for i in range(table_num):
            generated_ids.append([])
            for _ in range(len(feature_names_list[i])):
                generated_ids[i].append(list(range(num_embeddings[i] + OVER_COUNT)))
                random.shuffle(generated_ids[i][-1])
    dataset = dataset_class(batch_num, lookup_lens, num_embeddings, table_num, feature_names_list, generated_ids)
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    test_model = TestModel(rank, world_size, device, instances, feature_names_list, batch_num, collection_type)
    test_model.init_ddp_model(embedding_config, sharding_type, optim, lookup_lens)
    iter_ = iter(data_loader)
    module_list = getattr(test_model.module, collection_type)
    context = EmbCacheTrainPipelineContext(index=0, version=1)

    for i, module in enumerate(module_list):
        name = f"module.{i}"
        ctx = module.create_context()
        features = next(iter_).sparse_features
        context.input_dist_splits_requests[name] = module.input_dist(ctx, features)
        context.module_contexts[name] = ctx

    fuse_input_dist_splits(context)

    fused_splits_awaitables_dicts = []
    for fused_splits_awaitable in context.fused_splits_awaitables:
        output_lengths = fused_splits_awaitable.output_lengths
        lengths = fused_splits_awaitable.lengths
        if fused_splits_awaitable.splits_awaitable:
            splits_tensors = fused_splits_awaitable.splits_awaitable.wait()
        else:
            splits_tensors = []
        fused_splits_awaitables_dict = {
            "output_lengths": output_lengths,
            "lengths": lengths,
            "splits_tensors": splits_tensors,
        }
        fused_splits_awaitables_dicts.append(fused_splits_awaitables_dict)

    if not config["fname"].startswith("test_normal"):
        logging.debug("Skipping saving baseline for non-normal test: %s", config["fname"])
        return

    save_folder = os.path.join(TEST_ROOT_DIR, "configs", MODULE_NAME, "fuse_input_dist_splits")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    saved_file = os.path.join(save_folder, f"rank{rank}_{config['fname']}.pt")
    if not os.path.exists(saved_file):
        torch.save(fused_splits_awaitables_dicts, saved_file)
        logging.warning(
            "No baseline file found. This might be because you're running this test for the first time. "
            "The current output is being saved to the baseline file for future comparison. "
            "To perform accuracy checks, please re-run the test after the baseline file is generated."
        )
    else:
        base_line = torch.load(saved_file, weights_only=False)
        for obj1, obj2 in zip(base_line, fused_splits_awaitables_dicts):
            attributes_to_compare = ["output_lengths", "lengths", "splits_tensors"]
            assert are_features_equal(obj1, obj2, attributes_to_compare), \
                "Fused splits awaitables are not equal: {} != {}".format(obj1, obj2)
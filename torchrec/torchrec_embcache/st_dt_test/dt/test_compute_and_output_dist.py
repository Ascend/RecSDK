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
from torchrec_embcache.distributed.train_pipeline import (
    AwaitableAdapter,
    EmbcacheTrainPipelineContext,
)
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
    feature_names_lst = config["feature_names_lst"]
    instances = config.get("instances", 1)
    pool_type = getattr(torchrec.PoolingType, pool_type)
    collection_type = config["collection_type"]
    embedding_config = generate_hash_config(embedding_dims, num_embeddings, pool_type, feature_names_lst, 
                                            create_weight_init(init_fn), collection_type)
    generated_ids = []
    if dataset_class is BoundOutOfRangeRecDataset:
        for i in range(table_num):
            generated_ids.append([])
            for _ in range(len(feature_names_lst[i])):
                generated_ids[i].append(list(range(num_embeddings[i] + OVER_COUNT)))
                random.shuffle(generated_ids[i][-1])
    dataset = dataset_class(batch_num, lookup_lens, num_embeddings, table_num, feature_names_lst, generated_ids)
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    test_model = TestModel(rank, world_size, device, instances, feature_names_lst, batch_num, collection_type)
    test_model.init_ddp_model(embedding_config, sharding_type, optim, lookup_lens)
    iter_ = iter(data_loader)
    module_lst = getattr(test_model.module, collection_type)
    context = EmbcacheTrainPipelineContext(index=0, version=1)

    for i, module in enumerate(module_lst):
        name = f"module.{i}"
        ctx = module.create_context()
        features = next(iter_).sparse_features
        context.input_dist_splits_requests[name] = module.input_dist(ctx, features)
        context.module_contexts[name] = ctx

    fuse_input_dist_splits(context)

    kjt_list_dict = {}
    for names, awaitable in context.fused_splits_awaitables:
        for name, request in zip(names, awaitable.wait()):
            kjt_list_dict[name] = request.awaitables

    jt_list = []
    for i, module in enumerate(module_lst):
        name = f"module.{i}"
        kjt_lst = kjt_list_dict[name]
        post_waitable = module.post_input_dist(
            context.module_contexts[name], 
            kjt_lst
        )
        sparse_features = post_waitable.wait()
        swap_info_future = module.compute_swap_info_async(sparse_features)
        swap_info = swap_info_future.get()
        sparse_features[0].unique_indices = swap_info.batch_offs
        for j, sparse_feature in enumerate(sparse_features):
            sparse_features[j] = sparse_feature.to(test_model.npu_device, non_blocking=True)

        module_awaitable = module.compute_and_output_dist(ctx, sparse_features)
        jt = module_awaitable.wait()
        jt_list.append({"value": jt.values()})

    if not config["fname"].startswith("test_normal"):
        logging.debug("Skipping saving baseline for non-normal test: %s", config["fname"])
        return

    save_folder = os.path.join(TEST_ROOT_DIR, "configs", MODULE_NAME, "compute_and_output_dist")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    saved_file = os.path.join(save_folder, f"rank{rank}_{config['fname']}.pt")
    if not os.path.exists(saved_file):
        torch.save(jt_list, saved_file)
        logging.warning(
            "No baseline file found. This might be because you're running this test for the first time. "
            "The current output is being saved to the baseline file for future comparison. "
            "To perform accuracy checks, please re-run the test after the baseline file is generated."
        )
    else:
        base_line = torch.load(saved_file, weights_only=False)
        for obj1, obj2 in zip(base_line, jt_list):
            attributes_to_compare = ["value"]
            assert (
                are_features_equal(obj1, obj2, attributes_to_compare), 
                "jt values are not equal: {} != {}".format(obj1, obj2)
            )
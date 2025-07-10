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
    EmbCacheTrainPipelineContext,
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


def assert_kjt_equal(kjt1: KeyedJaggedTensor, kjt2: KeyedJaggedTensor, rtol=1e-5, atol=1e-5):
    """
    断言两个 KJT 对象在 keys、values、lengths、weights 上相等
    """
    assert kjt1.keys() == kjt2.keys(), f"Keys mismatch: {kjt1.keys()} vs {kjt2.keys()}"

    torch.testing.assert_close(
        kjt1.values(),
        kjt2.values(),
        rtol=rtol,
        atol=atol,
        msg=lambda s: f"Values mismatch: {s}"
    )

    torch.testing.assert_close(
        kjt1.lengths(),
        kjt2.lengths(),
        rtol=rtol,
        atol=atol,
        msg=lambda s: f"Lengths mismatch: {s}"
    )

    if kjt1.weights_or_none() is not None or kjt2.weights_or_none() is not None:
        torch.testing.assert_close(
            kjt1.weights(),
            kjt2.weights(),
            rtol=rtol,
            atol=atol,
            msg=lambda s: f"Weights mismatch: {s}"
        )


def assert_nested_kjt_equal(nested_kjt_list, nested_baseline_list, rtol=1e-5, atol=1e-5):
    """
    递归断言嵌套列表中的所有 KJT 相等
    """
    assert len(nested_kjt_list) == len(nested_baseline_list), \
        f"Outer list length mismatch: {len(nested_kjt_list)} vs {len(nested_baseline_list)}"

    for i, (kjt_sublist, baseline_sublist) in enumerate(zip(nested_kjt_list, nested_baseline_list)):
        assert isinstance(kjt_sublist, list) and isinstance(baseline_sublist, list), \
            f"Element at index {i} is not a list"

        assert len(kjt_sublist) == len(baseline_sublist), \
            f"Inner list length mismatch at index {i}: {len(kjt_sublist)} vs {len(baseline_sublist)}"

        for j, (kjt, baseline) in enumerate(zip(kjt_sublist, baseline_sublist)):
            assert isinstance(kjt, KeyedJaggedTensor) and isinstance(baseline, KeyedJaggedTensor), \
                f"Element at [{i}][{j}] is not a KeyedJaggedTensor"
            assert_kjt_equal(kjt, baseline, rtol=rtol, atol=atol)


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
    kjt_list = []
    for module in module_list:
        kjt_list.append([])
        ctx = module.create_context()
        features = next(iter_).sparse_features
        awaitable = module.input_dist(ctx, features)
        for awaitable in awaitable.awaitables:
            kjt = awaitable.wait().wait().wait()
            kjt_list[-1].append(kjt)

    if not config["fname"].startswith("test_normal"):
        logging.debug("Skipping baseline check for %s", config["fname"])
        return
    save_folder = os.path.join(TEST_ROOT_DIR, "configs", MODULE_NAME, "input_dist")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    saved_file = os.path.join(save_folder, f"rank{rank}_{config['fname']}.pt")
    if not os.path.exists(saved_file):
        torch.save(kjt_list, saved_file)
        logging.warning(
            "No baseline file found. This might be because you're running this test for the first time. "
            "The current output is being saved to the baseline file for future comparison. "
            "To perform accuracy checks, please re-run the test after the baseline file is generated."
        )
    else:
        base_line = torch.load(saved_file, weights_only=False)
        assert_nested_kjt_equal(kjt_list, base_line), "KJT lists are not equal. "


#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from collections import defaultdict
from typing import Callable

import pytz
import torch
import torch.multiprocessing as mp
import numpy as np
from dataset import (
    RandomRecDataset, 
    BoundOutOfRangeRecDataset, 
    FeatureNameNotInConfigRecDataset
)
from parse_configs import load_all_configs
from torch.autograd.profiler import record_function
from torch.optim import Adam, Adagrad
from torchrec_embcache.distributed.sharding.rw_sharding import EmbCacheRwSparseFeaturesDistAwaitable
from torchrec_embcache.distributed.utils import get_embedding_optim_num

from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable, 
    KJTListSplitsAwaitable, 
    KJTSplitsAllToAllMeta
)
from torchrec.distributed.train_pipeline.utils import TrainPipelineContext


OVER_COUNT = 10

TEST_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# utils for logging
def setup_logging(rank):
    from datetime import datetime

    this_time = str(
        datetime.now(tz=pytz.timezone("PRC")).strftime(
            "%m_%d_%H_%M_%S",
        )
    )
    format_message = logging.Formatter(
        fmt=f"[rank{rank}][%(levelname)s][%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    file_handler = logging.FileHandler(
        f"test_rank{rank}_{this_time}.log", encoding="utf-8"
    )
    file_handler.setFormatter(format_message)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


# utils for config checking
def is_bad_case(config):
    return config.get("is_bad_case", False)


def is_lookup_out_of_bound(config):
    if not is_bad_case(config):
        return False
    return config.get("RecDataset", "Random") == "BoundOutOfRange"


def feature_name_exists(config):
    if not is_bad_case(config):
        return False
    return config.get("RecDataset", "Random") == "FeatureNameNotInConfig"


def check_config(config):
    # 如果是dataloader越界，需要检查loop lookup_lens
    if config["RecDataset"] == "BoundOutOfRange":
        bound_out_of_range = False
        # if LOOP_TIMES*config["lookup_lens"]
        for i in range(config["table_num"]):
            if config["lookup_lens"] * config["BATCH_NUM"] > config["num_embeddings"][i] + OVER_COUNT:
                bound_out_of_range = True
                break
        if not bound_out_of_range:
            raise ValueError(
                "lookup_lens and BATCH_NUM is too small, if you want to test out of range, "
                "please set lookup_lens*BATCH_NUM*LOOP_TIMES > num_embeddings+OVER_COUNT"
                )
        
    # 需要检查是否超出显存
    # 表的大小，要考虑分表的情况：sum(instances*embedding_dim*num_embeddings)/WORLD_SIZE
    table_size = 0
    for embedding_dim, num_embedding in zip(config["embedding_dims"], config["num_embeddings"]):
        table_size += embedding_dim * num_embedding
    table_size = table_size * config["table_num"] / config["WORLD_SIZE"]
    # 查表的大小，lookup_lens*embedding_dim*len(feature_names)
    dtype_size = 4 # default fp32
    lookup_size = 0
    for embedding_dim, feature_names in zip(config["embedding_dims"], config["feature_names_list"]):
        lookup_size += embedding_dim * len(feature_names)
    total_size = (table_size + lookup_size) * dtype_size / (1024 * 1024)  # Convert to MB
    max_size = int(os.getenv("MAX_TABLE_SIZE_MB", 65536))
    if total_size > max_size:
        raise ValueError(
            "table size is too large, please reduce the table size"
            f" or increase the WORLD_SIZE, total_size: {total_size}, max_size: {max_size}"
        )

    # 需要检查device缓存是否够用
    multi_hot_sizes = [1] * config["table_num"]
    if config["optim"] == "Adagrad":
        embedding_optimizer_cls = torch.optim.Adagrad
    elif config["optim"] == "Adam":
        embedding_optimizer_cls = torch.optim.Adam
    else:
        raise ValueError(f"Unsupported optimizer: {config['optim']}")
    optim_num = get_embedding_optim_num(embedding_optimizer_cls)
    # 由于同时训练换入换出，最少要能放下2倍的batch_size的emb + optim
    weight_and_optim_count = optim_num + 1
    min_mem = np.sum(
        np.dot(
            np.multiply(config["embedding_dims"], multi_hot_sizes), 
            2 * dtype_size * config["lookup_lens"] * weight_and_optim_count
        )
    ) 
    max_device_mem_for_vectors = os.getenv("EMBCACHE_SIZE_ON_DEVICE_MEM")
    if not max_device_mem_for_vectors:
        raise EnvironmentError("EMBCACHE_SIZE_ON_DEVICE_MEM is not set, please set it in the environment")
    max_device_mem_for_vectors = int(max_device_mem_for_vectors)
    if max_device_mem_for_vectors < min_mem:
        raise ValueError(f"max_device_mem_for_vectors is not enough, \
            please increase the EMBCACHE_SIZE_ON_DEVICE_MEM or reduce the embedding_dim or lookup_lens, \
                current EMBCACHE_SIZE_ON_DEVICE_MEM: {max_device_mem_for_vectors}, min_mem: {min_mem}")


# utils for weight init
def create_weight_init(init_fn: Callable[[int], torch.Tensor]):
    """
    创建一个只接受 param 参数的 weight_init 函数。
    
    Args:
        init_fn: 一个根据输入维度返回初始化张量的函数。
        
    Returns:
        一个新的 weight_init 函数，它只接受 param 参数。
    """
    def weight_init(param: torch.nn.Parameter):
        if len(param.shape) != 2:
            return
        in_dim = param.shape[1]
        torch.manual_seed(in_dim)
        result = init_fn(in_dim).repeat(param.shape[0], 1)
        param.data.copy_(result)
    return weight_init


# 初始化器定义
def init_random(in_dim):
    return torch.randn((1, in_dim))


def init_linspace(in_dim):
    return torch.linspace(0, 1, steps=in_dim).unsqueeze(0)


def init_ones(in_dim):
    return torch.ones((1, in_dim))


def init_zeros(in_dim):
    return torch.zeros((1, in_dim))


def init_uniform(in_dim):
    return torch.empty((1, in_dim)).uniform_()


# utils for conftest
def generate_test_cases(metafunc, all_configs):
    """
    Pytest hook to generate tests dynamically based on the configurations.
    """
    test_case_name = metafunc.function.__name__
    configs_for_case = all_configs.get(test_case_name, [])

    # 获取命令行参数
    config_file = metafunc.config.getoption("--test-config-file")

    if config_file:
        # 如果指定了单个配置文件，则只使用该配置
        config_file = config_file.split(".")[0].split("/")[-1]  # Remove the file extension
        # Find the matching config for the specified file
        selected_config = next((cfg for fname, cfg in configs_for_case if fname == config_file), None)
        if selected_config is not None:
            metafunc.parametrize("config", [selected_config], ids=[config_file])

    elif configs_for_case:
        metafunc.parametrize(
            "config", 
            [cfg for _, cfg in configs_for_case], ids=[fname for fname, _ in configs_for_case]
        )
    else:
        metafunc.parametrize("config", [], scope="function")


def get_all_configs(module_name: str) -> dict:
    config_dir = os.path.join(TEST_ROOT_DIR, "configs", module_name)

    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory {config_dir} does not exist.")
    all_configs = load_all_configs(config_dir)
    return all_configs


# utils for dt test
class TestFusedKJTListSplitsAwaitable(FusedKJTListSplitsAwaitable):

    @property
    def lengths(self):
        return self._lengths
    
    @property
    def splits_awaitables(self):
        return self._splits_awaitables
    
    @property
    def splits_awaitable(self):
        return self._splits_awaitable


def fuse_input_dist_splits(context: TrainPipelineContext) -> None:
    with record_function("## _fuse_input_dist_splits ##"):
        names_per_pg = defaultdict(list)
        for name, request in context.input_dist_splits_requests.items():
            pg = None
            if isinstance(request, KJTListSplitsAwaitable):
                for awaitable in request.awaitables:
                    if isinstance(awaitable, KJTSplitsAllToAllMeta) or isinstance(
                        awaitable, EmbCacheRwSparseFeaturesDistAwaitable
                    ):
                        pg = awaitable.pg
                        break
            names_per_pg[pg].append(name)

        for name, request in context.input_dist_splits_requests.items():
            for ind, awaitable in enumerate(
                context.input_dist_splits_requests[name].awaitables
            ):
                if isinstance(awaitable, EmbCacheRwSparseFeaturesDistAwaitable):
                    context.input_dist_splits_requests[name].awaitables[ind] = \
                        context.input_dist_splits_requests[name].awaitables[ind].wait()

        for pg, names in names_per_pg.items():
            context.fused_splits_awaitables.append(
                (
                    names,
                    TestFusedKJTListSplitsAwaitable(
                        # pyre-ignore[6]
                        requests=[
                            context.input_dist_splits_requests[name] for name in names
                        ],
                        contexts=[
                            (
                                context.module_contexts_next_batch[name]
                                if context.version == 0
                                else context.module_contexts[name]
                            )
                            for name in names
                        ],
                        pg=pg,
                    ),
                )
            )


# utils for public test
def run_model_with_config(config, execute_func):
    if config.get("device", "npu") == "cpu" and config.get("sharding_type", "table_wise") == "row_wise":
        return
    mp.spawn(
        execute_func,
        args=(config,),
        nprocs=config.get("WORLD_SIZE", 2),
        join=True,
    )


def compare_tensors(tensor1, tensor2):
    if tensor1 is None and tensor2 is None:
        return True
    if tensor1 is None or tensor2 is None:
        return False
    return torch.allclose(tensor1, tensor2)


def compare_list(list1, list2):
    if list1 is None and list2 is None:
        return True
    if list1 is None or list2 is None:
        return False
    if len(list1) != len(list2):
        return False
    
    for item1, item2 in zip(list1, list2):
        if isinstance(item1, list) and isinstance(item2, list):
            if not compare_list(item1, item2):
                return False
        elif isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
            if not compare_tensors(item1, item2):
                return False
        elif item1 != item2:
            return False
    return True


def are_features_equal(obj1, obj2, attributes_to_compare):
    for attr in attributes_to_compare:
        value1 = obj1.get(attr, None)
        value2 = obj2.get(attr, None)

        if value1 is None or value2 is None:
            logging.error(f"Attribute '{attr}' not found in one of the objects.")
            return False
        elif isinstance(value1, list):
            if not compare_list(value1, value2):
                logging.debug("Lists are not equal: %s != %s", value1, value2)
                return False
        elif isinstance(value1, torch.Tensor):
            if not compare_tensors(value1, value2):
                logging.debug("Tensors are not equal: %s != %s", value1, value2)
                return False
    
    return True


DATASET_REGISTRY = {
    "RandomRecDataset": RandomRecDataset,
    "BoundOutOfRangeRecDataset": BoundOutOfRangeRecDataset,
    "FeatureNameNotInConfigRecDataset": FeatureNameNotInConfigRecDataset,
}


INIT_FN_REGISTRY = {
    "init_random": init_random,
    "init_linspace": init_linspace,
    "init_ones": init_ones,
    "init_zeros": init_zeros,
    "init_uniform": init_uniform,
}


OPTIM_REGISTRY = {
    "Adagrad": Adagrad,
    "Adam": Adam,
}
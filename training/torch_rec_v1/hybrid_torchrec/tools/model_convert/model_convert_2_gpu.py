#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Any, Union, LiteralString, Type

import numpy as np
import torch.nn
from torch.optim import Adam, Adagrad, SGD
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection

SAVE_PATH_MAX_LEN = 1024
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_MAX_RECURSIVE_TIMES = 500
_MAX_LOOP_TIMES = 500
_FLOAT32_BYTES = 4
_INT64_BYTES = 8
_OPEN_DIR_MODE = 0o750
_OPEN_FILE_FLAG = os.O_WRONLY | os.O_CREAT
_OPEN_FILE_MODE = 0o640
_UNSUPPORTED_FILE_MODE_MASK = 0o022
_READ_FILE_FLAG = os.O_RDONLY
_ATTRIBUTE_EMB_DIM_INDEX = 2
_SLICE_ATTRIBUTE = "slice.attribute"
_SLICE_DATA = "slice.data"
# key的slice.attribute中数据长度为2, emb/momentum slice.attribute中数据长度为3
_ATTRIBUTE_MIN_LEN = 2


logging.getLogger().setLevel(logging.INFO)

_OPTIMIZER_NUM_DICT = {
    Adam: 2,
    Adagrad: 1,
    SGD: 0,
}


def check_path(path: str) -> None:
    p = Path(path)
    if not p.is_file() and not p.is_symlink():
        raise ValueError(f"file not exist or file is symlink, file: {path}")


def safe_makedirs(path: Union[str, Path]):
    if not os.path.exists(path):
        os.makedirs(path, mode=_OPEN_DIR_MODE, exist_ok=True)


def write_binary_data(writing_path: Union[str, LiteralString, bytes], file_name: str, data: np.ndarray):
    safe_makedirs(writing_path)
    target_file = os.path.join(writing_path, file_name)
    with os.fdopen(os.open(target_file, _OPEN_FILE_FLAG, _OPEN_FILE_MODE), "wb") as file:
        file.write(data.tobytes())


def read_binary_data(reading_path: str) -> tuple[list[Any], list[Any]]:
    data_file, attribute_file = _SLICE_DATA, _SLICE_ATTRIBUTE
    data_file = os.path.join(reading_path, data_file)
    attribute_file = os.path.join(reading_path, attribute_file)
    if not os.path.exists(attribute_file):
        raise FileExistsError(f"attribute file:{attribute_file} does not exist when reading.")

    attributes = read_attribute_file(attribute_file)
    attributes = list(attributes)
    if len(attributes) < _ATTRIBUTE_MIN_LEN or attributes[1] <= 0:
        return attributes, []

    if not os.path.exists(data_file):
        raise FileExistsError(f"data file {data_file} does not exist when reading.")
    data_to_restore = read_data_file(data_file)
    return attributes, data_to_restore


def read_attribute_file(target_file: str):
    with os.fdopen(os.open(target_file, _READ_FILE_FLAG, _OPEN_FILE_MODE), "rb") as fin:
        check_path(target_file)
        attributes = fin.read()
        try:
            attributes = np.frombuffer(attributes, dtype=np.int64)
        except ValueError as err:
            raise RuntimeError(f"get attributes from file {target_file} failed.") from err
    logging.info("read attribute file data: %s, file: %s", attributes, target_file)
    return attributes


def read_data_file(target_file: str):
    data_type = np.int64 if "/key/" in target_file else np.float32
    with os.fdopen(os.open(target_file, _READ_FILE_FLAG, _OPEN_FILE_MODE), "rb") as file:
        check_path(target_file)
        data_to_restore = np.fromfile(target_file, dtype=data_type)
    logging.info("read detail data file shape: %s, file: %s", data_to_restore.shape, target_file)
    return data_to_restore


def _get_data_dim(attribute_data: Union[np.array, List[int]], file_path: str):
    if len(attribute_data) < _ATTRIBUTE_MIN_LEN:
        raise ValueError(f"attribute data must have at least 2 elements, maybe file has been tampered,"
                         f" file:{file_path}")
    return attribute_data[-1]


class ModelConverter:
    def __init__(self, rank: int):
        self.rank = rank
        self.optimizer_num = 0
        self.modules: List[ShardedEmbeddingBagCollection] = []

    @staticmethod
    def is_timestamp_format(dir_name: str) -> bool:
        try:
            datetime.strptime(dir_name, TIMESTAMP_FORMAT)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_latest_load_path(path):
        path = Path(path)
        dirs = [d for d in path.iterdir() if d.is_dir() and ModelConverter.is_timestamp_format(d.name)]
        if not dirs:
            raise ValueError(f"expect a timestamp directory but empty in path")
        latest_dir = max(d.name for d in dirs)
        return os.path.join(os.path.realpath(path), latest_dir)

    def load(
            self,
            model: torch.nn.Module,
            path: str,
            optimizer_type: Union[Type[Adam], Type[Adagrad], Type[SGD]]
    ):
        """
        读取NPU已保存的模型数据，直接更新到传入的分片后的模型中

        Args:
            model (torch.nn.Module): DistributedModelParallel或者ShardedEmbeddingBagCollection
            path (str): NPU sparse Embedding保存路径. 相对路径和绝对路径均可
            optimizer_type (Union[Adam, Adagrad, SGD]): 稀疏表使用的优化器类型 class

        """
        self.modules.clear()
        self._find_all_sharded_module_instance(model)
        self._check_module_instance_len()
        path = os.path.realpath(path)
        path = self.get_latest_load_path(path)
        if optimizer_type not in _OPTIMIZER_NUM_DICT.keys():
            raise ValueError(f"current optimizer type: {optimizer_type} is not supported,"
                             f" it must be in: {_OPTIMIZER_NUM_DICT.keys()}")
        self.optimizer_num = _OPTIMIZER_NUM_DICT[optimizer_type]

        # 加载数据
        for mod in self.modules:
            self._load_emb_and_optimizer(mod, path)

    def _load_emb_and_optimizer(self, mod: ShardedEmbeddingBagCollection, path: str):
        for table_index, table_config in enumerate(mod._embedding_bag_configs):
            table_name = table_config.name
            key_path = self._get_key_data_dir(path, table_name)
            # 1 加载 original_ids
            key_attribute, original_ids = read_binary_data(key_path)
            if len(original_ids) == 0:
                logging.info("table name:%s, current table does not have any ids when load.", table_name)
                continue

            # 2 加载 embedding
            original_ids = self._load_embedding_data(mod, original_ids, path, table_config)

            # 3 加载 optimizer momentum
            if not self.optimizer_num:
                if table_index == 0:
                    logging.info("The optimizer does not have momentum info, skip.")
                continue
            fused_opt = mod.fused_optimizer
            self._load_optimizer_data(table_index, table_config, fused_opt, original_ids, path)

        logging.debug("In load, rank:%d, load all ids, embedding and optimizer data end!", self.rank)

    def _load_embedding_data(self, mod, original_ids, path, table_config):
        emb_path = self._get_emb_data_dir(path, table_config.name)
        emb_attribute, emb_data = read_binary_data(emb_path)
        embedding_dim = emb_attribute[_ATTRIBUTE_EMB_DIM_INDEX]
        if table_config.embedding_dim != embedding_dim:
            raise ValueError(f"table:{table_config.name} dim:{table_config.embedding_dim} is not equal to "
                             f"saved embedding dim:{embedding_dim}")

        weights: torch.Tensor = mod.embedding_bags[table_config.name].weight.data.reshape(-1, embedding_dim)
        device = weights.device
        original_ids = torch.tensor(original_ids, dtype=torch.long, device=device)
        emb_data = torch.tensor(emb_data, dtype=torch.float32, device=device).reshape(-1, embedding_dim)
        weights.index_put_([original_ids], emb_data)
        logging.debug("In load, rank:%d, table name:%s, load embedding data end.",
                      self.rank, table_config.name)
        return original_ids

    def _load_optimizer_data(self, table_index, config, fused_opt, original_ids, path):
        table_name = config.name
        meta_data = fused_opt.param_groups[0]["params"][table_index]
        momentum_dict = fused_opt.state[meta_data]
        # 遍历每个momentum
        for momentum_index, momentum_key in enumerate(momentum_dict.keys()):
            if (table_name + ".") not in momentum_key:
                raise ValueError(f"momentum key:{momentum_key} is not match for table name {table_name}")
            momentum_path = self._get_momentum_data_dir(momentum_index, path, table_name)
            momentum_attribute, momentum_data = read_binary_data(momentum_path)
            load_data_dim = _get_data_dim(momentum_attribute, momentum_path)
            if load_data_dim != config.embedding_dim:
                raise ValueError(f"momentum key: {momentum_key} loaded data dim:{load_data_dim} "
                                 f"is not equal to table config dim:{config.embedding_dim}")
            momentum_data_opt = momentum_dict[momentum_key].local_shards()[0].tensor
            device = momentum_data_opt.device
            momentum_data = torch.tensor(momentum_data, dtype=torch.float32, device=device)
            momentum_data = momentum_data.reshape(-1, load_data_dim)
            momentum_data_opt.index_put_([original_ids], momentum_data)
            logging.debug("In load, rank:%d, table name:%s, momentum_key:%s, load data end.",
                          self.rank, table_name, momentum_key)

    def _get_key_data_dir(self, path, table_name):
        return os.path.join(path, table_name, f"rank{self.rank}", "key")

    def _get_emb_data_dir(self, path, table_name):
        return os.path.join(path, table_name, f"rank{self.rank}", "embedding")

    def _get_momentum_data_dir(self, ind: int, path, table_name):
        return os.path.join(path, table_name, f"rank{self.rank}", f"momentum{ind + 1}")

    def _find_all_sharded_module_instance(self, module: ShardedEmbeddingBagCollection,
                                          this_recur_step: int = 0):
        if this_recur_step >= _MAX_RECURSIVE_TIMES:
            raise RuntimeError(f"Recursion depth not greater than {_MAX_RECURSIVE_TIMES}")
        for ind, (_, child) in enumerate(module.named_children()):
            if ind >= _MAX_LOOP_TIMES:
                raise RuntimeError(f"Len of module children should not be greater than {_MAX_LOOP_TIMES}")
            if isinstance(child, ShardedEmbeddingBagCollection):
                self.modules.append(child)
            self._find_all_sharded_module_instance(child, this_recur_step + 1)

    def _check_module_instance_len(self):
        if len(self.modules) == 0:
            raise ValueError("param `module` must has at least one child module which "
                             "type is ShardedEmbeddingBagCollection.")

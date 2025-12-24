#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Any, Union

import numpy as np
import torch.distributed as dist
import torch.nn

from hybrid_torchrec.distributed.batched_embedding_kernel import (
    HybridSplitTableBatchedEmbeddingBagsCodegen,
)
from hybrid_torchrec.distributed.hash_embeddingbag import HybridShardedHashEmbeddingBagCollection
from hybrid_torchrec.utils import check_path, safe_makedirs

SAVE_PATH_MAX_LEN = 1024
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_MAX_RECURSIVE_TIMES = 500
_MAX_LOOP_TIMES = 500
_FLOAT32_BYTES = 4
_INT64_BYTES = 8
_OPEN_FILE_FLAG = os.O_WRONLY | os.O_CREAT
_OPEN_FILE_MODE = 0o640
_UNSUPPORTED_FILE_MODE_MASK = 0o022
_READ_FILE_FLAG = os.O_RDONLY
_HASH_START_INDEX = 0
_ATTRIBUTE_EMB_DIM_INDEX = 2
_SLICE_ATTRIBUTE = "slice.attribute"
_SLICE_DATA = "slice.data"
# key的slice.attribute中数据长度为2, emb/momentum slice.attribute中数据长度为3
_ATTRIBUTE_MIN_LEN = 2


def write_binary_data(writing_path, file_name: str, data: np.ndarray):
    safe_makedirs(writing_path)
    target_file = os.path.join(writing_path, file_name)
    with os.fdopen(os.open(target_file, _OPEN_FILE_FLAG, _OPEN_FILE_MODE), "wb") as file:
        file.write(data.tobytes())


def read_binary_data(reading_path) -> tuple[list[Any], list[Any]]:
    data_file, attribute_file = _SLICE_DATA, _SLICE_ATTRIBUTE
    data_file = os.path.join(reading_path, data_file)
    attribute_file = os.path.join(reading_path, attribute_file)
    if not os.path.exists(attribute_file):
        raise FileNotFoundError(f"attribute file:{attribute_file} does not exist when reading.")

    attributes = read_attribute_file(attribute_file)
    attributes = list(attributes)
    if len(attributes) < _ATTRIBUTE_MIN_LEN or attributes[1] <= 0:
        return attributes, []

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"data file {data_file} does not exist when reading.")
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
    logging.info("Read attribute file data: %s, file: %s", attributes, target_file)
    return attributes


def read_data_file(target_file: str):
    data_type = np.int64 if "/key/" in target_file else np.float32
    with os.fdopen(os.open(target_file, _READ_FILE_FLAG, _OPEN_FILE_MODE), "rb") as file:
        check_path(target_file)
        data_to_restore = np.fromfile(target_file, dtype=data_type)
    logging.info("Read detail data file shape: %s, file: %s", data_to_restore.shape, target_file)
    return data_to_restore


def _get_data_dim(attribute_data: Union[np.array, List[int]], file_path: str):
    if len(attribute_data) < _ATTRIBUTE_MIN_LEN:
        raise ValueError(f"attribute data must have at least 2 elements, maybe file has been tampered,"
                         f" file:{file_path}")
    return attribute_data[-1]


class Saver:
    def __init__(self, rank: int = None):
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()
                logging.warning("Param rank id is None and distributed model has been initialized,"
                                " get rank by dist.get_rank() is:%d", rank)
            else:
                raise ValueError("param `rank` must not be None when torch.distributed.is_initialized() is False.")
        else:
            if not (isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0):
                raise ValueError("param rank must be an integer and need greater or equal than 0.")
            if dist.is_initialized():
                world_size = torch.distributed.get_world_size()
                if rank >= world_size:
                    raise ValueError(f"param `rank` must less than torch distribution world_size:{world_size},"
                                     f" but got rank {rank}")
        self.rank: int = rank
        self.modules = []

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
        dirs = [d for d in path.iterdir() if d.is_dir() and Saver.is_timestamp_format(d.name)]
        if not dirs:
            raise ValueError(f"expect a timestamp directory but empty in path")
        latest_dir = max(d.name for d in dirs)
        return os.path.join(os.path.realpath(path), latest_dir)

    @staticmethod
    def _get_format_path():
        return datetime.now(tz=timezone.utc).strftime(TIMESTAMP_FORMAT)

    def save(self, module: torch.nn.Module, path: str) -> None:
        """
        保存纯显存模式稀疏表数据接口
        保存数据格式为：
            {path}/{table_name}/rank{rank_id}/
                |--key
                    |--slice.attribute  # 详细数据为：[data_type_bytes, key_shape], 如：[8, 20000]
                    |--slice.data  # id list
                |--embedding
                    |--slice.attribute  # 详细数据为：[data_type_bytes, flat_embedding_shape], 如：[4, 20000, 16]
                    |--slice.data  # flat embedding数据
                |--momentum1
                    |--slice.attribute  # 同embedding 'slice.attribute'
                    |--slice.data  # flat momentum 数据
                |--momentum2
                    |--slice.attribute  # 同embedding 'slice.attribute'
                    |--slice.data  # flat momentum 数据

        Args:
            module (torch.nn.Module): 实际传入HybridShardedHashEmbeddingBagCollection对象，
                或经过torchrec.distributed.DistributedModelParallel分表后的dmp_model
            path (str): 稀疏表保存路径

        Note: ids长度为空时，不会保存slice.data文件
        """
        start_time = time.time()
        check_path(path)
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"param `module` must an instance of torch.nn.Module, but got:{type(module)}")

        if not dist.is_initialized():
            raise ValueError("when save, the status of torch.distributed.is_initialized() must be True, but got False.")
        path = os.path.realpath(path)

        self.modules.clear()
        self._find_all_sharded_module_instance(module)
        self._check_module_instance_len()
        safe_makedirs(path)
        logging.info("In save scene, path:%s, module info:%s", path, self.modules)
        for mod in self.modules:
            mod: HybridShardedHashEmbeddingBagCollection
            logging.info("In save scene, module info:%s", mod)
            self._save_emb_and_optimizer(path, mod)
        end_time = time.time()
        logging.info("In save, rank:%d, save sparse data cost time: %.8f(s).",
                     self.rank, end_time - start_time)

    def load(self, module: torch.nn.Module, path: str) -> None:
        """
        加载纯显存模式稀疏表数据接口

        Args:
            module (torch.nn.Module): 实际传入HybridShardedHashEmbeddingBagCollection对象，
                或经过torchrec.distributed.DistributedModelParallel分表后的dmp_model
            path (str): 已保存稀疏表的路径。参数和`save`接口的path参数相同即可
        """
        start_time = time.time()
        check_path(path, need_exist=True, is_dir=True)
        self.modules.clear()
        self._find_all_sharded_module_instance(module)
        self._check_module_instance_len()
        path = os.path.realpath(path)
        check_path(path)
        for mod in self.modules:
            self._load_emb_and_optimizer(path, mod)
        logging.info("In load, rank:%d, load sparse data cost time: %.8f(s).",
                     self.rank, time.time() - start_time)

    def _find_all_sharded_module_instance(self, module: torch.nn.Module,
                                          this_recur_step: int = 0):
        if this_recur_step >= _MAX_RECURSIVE_TIMES:
            raise RuntimeError(f"Recursion depth not greater than {_MAX_RECURSIVE_TIMES}")
        for ind, (_, child) in enumerate(module.named_children()):
            if ind >= _MAX_LOOP_TIMES:
                raise RuntimeError(f"length of module children should not be greater than {_MAX_LOOP_TIMES}")
            if isinstance(child, HybridShardedHashEmbeddingBagCollection):
                self.modules.append(child)
            self._find_all_sharded_module_instance(child, this_recur_step + 1)

    def _check_module_instance_len(self):
        if len(self.modules) == 0:
            raise ValueError("param `module` must has at least one child module which "
                             "type is HybridShardedHashEmbeddingBagCollection.")

    def _save_emb_and_optimizer(self, path, mod: HybridShardedHashEmbeddingBagCollection):
        # 优化器参数个数
        for table_index, table_config in enumerate(mod.get_embedding_bag_configs()):
            table_name = table_config.name
            ids_mapper = mod.table2hashmap[table_name]
            original_ids = ids_mapper.export_ids_and_indices()
            original_ids = original_ids.numpy()
            codegen: HybridSplitTableBatchedEmbeddingBagsCodegen = mod.get_batched_embedding_kernels()[0][0]
            optimizer_num = codegen.get_optimizer_num()
            # 1 保存key attribute 和 data
            key_path = self._get_key_data_dir(path, table_name)
            key_attribute_data = list(original_ids.shape)
            key_attribute_data.insert(0, _INT64_BYTES)
            write_binary_data(key_path, _SLICE_ATTRIBUTE, np.array(key_attribute_data).astype(np.int64))
            if len(original_ids) == 0:
                self._write_empty_attribute_file(path, table_name, table_config.embedding_dim, optimizer_num)
                logging.info("The table name:%s, current table does not have any ids when save.", table_name)
                continue
            original_ids_num = len(original_ids) + _HASH_START_INDEX
            logging.info("In save, rank:%d, table name:%s, original ids data shape:%s",
                         self.rank, table_name, len(original_ids))
            write_binary_data(key_path, _SLICE_DATA, np.array(original_ids).astype(np.int64))

            # 2 保存emb attribute 和 data
            self._save_embedding_data(mod, original_ids_num, path, table_config)

            # 3 保存优化器 attribute 和 data
            if optimizer_num == 0:
                logging.info("The optimizer does not have momentum info, skip.")
                continue
            fused_opt = mod.fused_optimizer
            self._save_optimizer_data(table_index, table_config, fused_opt, original_ids_num, path)

    def _save_optimizer_data(self, table_index, table_config, fused_opt, original_ids_num, path):
        table_name = table_config.name
        meta_data = fused_opt.param_groups[0]["params"][table_index]
        momentum_dict = fused_opt.state[meta_data]
        for momentum_index, momentum_key in enumerate(momentum_dict.keys()):
            if (table_name + ".") not in momentum_key:
                raise ValueError(f"Momentum key {momentum_key} is not match for table name {table_name}")
            momentum_path = self._get_momentum_data_dir(momentum_index, path, table_name)
            momentum_data = momentum_dict[momentum_key].local_shards()[0].tensor
            momentum_data = momentum_data[_HASH_START_INDEX: original_ids_num, :]
            momentum_attribute_data = [_FLOAT32_BYTES]
            momentum_attribute_data.extend(momentum_data.shape)
            logging.info("In save, rank:%d, table name:%s, momentum data key:%s, momentum data shape: %s",
                         self.rank, table_name, momentum_key, momentum_data.shape)
            write_binary_data(momentum_path, _SLICE_ATTRIBUTE, np.array(momentum_attribute_data).astype(np.int64))
            write_binary_data(momentum_path, _SLICE_DATA, momentum_data.cpu().numpy())

    def _save_embedding_data(self, mod, original_ids_num, path, table_config):
        weights = mod.embedding_bags[table_config.name].weight.data.to("cpu")
        weights = weights.reshape(-1, table_config.embedding_dim)
        emb_path = self._get_emb_data_dir(path, table_config.name)
        emb_attribute_data = [_FLOAT32_BYTES]
        if len(weights) > (original_ids_num - _HASH_START_INDEX):
            weights = weights[_HASH_START_INDEX:original_ids_num, :]
        emb_attribute_data.extend(weights.shape)
        logging.info("In save, rank:%d, table name:%s, embedding data shape:%s",
                     self.rank, table_config.name, weights.shape)
        write_binary_data(emb_path, _SLICE_ATTRIBUTE, np.array(emb_attribute_data).astype(np.int64))
        write_binary_data(emb_path, _SLICE_DATA, weights.numpy())

    def _load_emb_and_optimizer(self, path: str, mod: HybridShardedHashEmbeddingBagCollection):
        device = mod.get_device()
        for table_index, table_config in enumerate(mod.get_embedding_bag_configs()):
            table_name = table_config.name
            key_path = self._get_key_data_dir(path, table_name)
            key_attribute, original_ids = read_binary_data(key_path)
            if len(original_ids) == 0:
                logging.info("The table name:%s, current table does not have any ids when load.", table_name)
                continue

            # 1 加载 original_ids 到IdsMapper
            ids_mapper = mod.table2hashmap[table_name]
            original_ids = torch.tensor(original_ids, dtype=torch.int64).reshape(-1)
            ids_mapper.load_original_ids(original_ids)
            logging.info("In load, rank:%d, table name:%s, load ids mapper end.", self.rank, table_name)

            # 2 加载 embedding
            original_ids_num = len(original_ids) + _HASH_START_INDEX
            self._load_embedding_data(device, mod, original_ids_num, path, table_config)

            # 3 加载 optimizer momentum
            codegen: HybridSplitTableBatchedEmbeddingBagsCodegen = mod.get_batched_embedding_kernels()[0][0]
            optimizer_num = codegen.get_optimizer_num()
            if not optimizer_num:
                if table_index == 0:
                    logging.info("The optimizer does not have momentum info, skip.")
                continue
            fused_opt = mod.fused_optimizer
            self._load_optimizer_data(table_index, table_config, fused_opt, original_ids_num, path)

        logging.info("In load, rank:%d, load all ids, embedding and optimizer data end!", self.rank)

    def _load_embedding_data(self, device, mod, original_ids_num, path, table_config):
        emb_path = self._get_emb_data_dir(path, table_config.name)
        emb_attribute, emb_data = read_binary_data(emb_path)
        embedding_dim = emb_attribute[_ATTRIBUTE_EMB_DIM_INDEX]
        if table_config.embedding_dim != embedding_dim:
            raise ValueError(f"table:{table_config.name} dim:{table_config.embedding_dim} is not equal to "
                             f"saved embedding dim:{embedding_dim}")

        hash_indices = torch.arange(_HASH_START_INDEX, original_ids_num, dtype=torch.long, device=device)
        emb_data = torch.tensor(emb_data, dtype=torch.float32, device=device).reshape(-1, embedding_dim)
        weights: torch.Tensor = mod.embedding_bags[table_config.name].weight.data.reshape(-1, embedding_dim)
        weights.index_put_([hash_indices], emb_data)
        logging.info("In load, rank:%d, table name:%s, load embedding data end.",
                     self.rank, table_config.name)

    def _load_optimizer_data(self, table_index, table_config, fused_opt, original_ids_num, path):
        table_name = table_config.name
        meta_data = fused_opt.param_groups[0]["params"][table_index]
        momentum_dict = fused_opt.state[meta_data]
        # 遍历每个momentum
        for momentum_index, momentum_key in enumerate(momentum_dict.keys()):
            if (table_name + ".") not in momentum_key:
                raise ValueError(f"Momentum key:{momentum_key} is not match for table name {table_name}")
            momentum_path = self._get_momentum_data_dir(momentum_index, path, table_name)
            momentum_attribute, momentum_data = read_binary_data(momentum_path)
            load_data_dim = _get_data_dim(momentum_attribute, momentum_path)
            if load_data_dim != table_config.embedding_dim:
                raise ValueError(f"Momentum key: {momentum_key} loaded data dim:{load_data_dim} "
                                 f"is not equal to table config dim:{table_config.embedding_dim}")
            momentum_data_opt = momentum_dict[momentum_key].local_shards()[0].tensor
            device = momentum_data_opt.device
            momentum_data = torch.tensor(momentum_data, dtype=torch.float32, device=device)
            momentum_data = momentum_data.reshape(-1, load_data_dim)
            hash_indices = torch.arange(_HASH_START_INDEX, original_ids_num, dtype=torch.long, device=device)
            momentum_data_opt.index_put_([hash_indices], momentum_data)
            logging.info("In load, rank:%d, table name:%s, momentum_key:%s, load data end.",
                         self.rank, table_name, momentum_key)

    def _get_key_data_dir(self, path, table_name):
        return os.path.join(path, table_name, f"rank{self.rank}", "key")

    def _get_emb_data_dir(self, path, table_name):
        return os.path.join(path, table_name, f"rank{self.rank}", "embedding")

    def _get_momentum_data_dir(self, ind: int, path, table_name):
        return os.path.join(path, table_name, f"rank{self.rank}", f"momentum{ind + 1}")

    def _write_empty_attribute_file(self, path, table_name, embedding_dim, optimizer_num):
        emb_path = self._get_emb_data_dir(path, table_name)
        attribute_data = np.array([_FLOAT32_BYTES, 0, embedding_dim]).astype(np.int64)
        write_binary_data(emb_path, _SLICE_ATTRIBUTE, attribute_data)

        for i in range(optimizer_num):
            momentum_path = self._get_momentum_data_dir(i, path, table_name)
            write_binary_data(momentum_path, _SLICE_ATTRIBUTE, attribute_data)

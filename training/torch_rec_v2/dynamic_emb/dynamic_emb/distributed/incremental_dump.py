#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from dynamic_emb.distributed.dump_load import find_sharded_modules, get_dynamic_emb_module


def is_valid_score_threshold(score_threshold: Any) -> bool:
    """
    Check if score_threshold is instance of Dict[str, Dict[str, int]].
    """
    if not isinstance(score_threshold, dict):
        return False

    for key, value in score_threshold.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, dict):
            return False

        for inner_key, inner_value in value.items():
            if not isinstance(inner_key, str):
                return False
            if not isinstance(inner_value, int):
                return False

    return True


def set_score(
        model: torch.nn.Module, table_score: Union[int, Dict[str, Dict[str, int]]]
) -> None:
    """Set score for dynamic embedding tables in the model.

    NOTE: This function is adapted from the NVIDIA implementation to match
    the interfaces provided in this project. Only per-table score setting
    is supported; incremental dump based on score is not implemented on NPU.

    Args:
        model (torch.nn.Module): Model that contains sharded dynamic embedding
            collections.
        table_score (Union[int, Dict[str, Dict[str, int]]]): Score setting
            strategy.
            - If `int`, apply the same score to all dynamic embedding tables.
            - If `Dict[str, Dict[str, int]]`, set scores by collection path and
              table name. The outer key is collection path in model, and the
              inner key is dynamic embedding table name.

    Returns:
        None

    Raises:
        ValueError: If `table_score` is neither `int` nor
            `Dict[str, Dict[str, int]]`.
    """
    if isinstance(table_score, int):
        set_all_table = True
    elif is_valid_score_threshold(table_score):
        set_all_table = False
    else:
        raise ValueError("DynamicEmb Error: table_score should be int or Dict[str, Dict[str, int]]")

    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any ShardedDynamicEmbeddingCollection "
            "or ShardedDynamicEmbeddingBagCollection module, can't set score!",
            UserWarning,
        )
        return

    check_dynamic_emb_modules_lists: List[List[nn.Module]] = []
    for _, _, tmp_collection_module in collections_list:
        check_dynamic_emb_modules_lists.append(get_dynamic_emb_module(tmp_collection_module))

    has_dynamic_emb = any(len(m_list) > 0 for m_list in check_dynamic_emb_modules_lists)
    if not has_dynamic_emb:
        warnings.warn(
            "Input model don't have any Dynamic embedding tables, can't set score!",
            UserWarning,
        )
        return

    if not set_all_table:
        collection_names_in_module = set()
        filtered_collections_list: List[Tuple[str, str, nn.Module]] = []

        for module_path, module_name, module in collections_list:
            # 本项目中，dump/load 等接口使用 module_path 作为唯一标识
            collection_names_in_module.add(module_path)
            if module_path in table_score.keys():
                filtered_collections_list.append((module_path, module_name, module))

        collections_list = filtered_collections_list

        for input_collection_name in table_score.keys():
            if input_collection_name not in collection_names_in_module:
                warnings.warn(
                    f"sharded module '{input_collection_name}' specified in table_score not found in the model",
                    UserWarning,
                )

    for collection_path, _, collection_module in collections_list:
        dynamic_emb_modules = get_dynamic_emb_module(collection_module)
        for dynamic_emb_module in dynamic_emb_modules:
            table_names = dynamic_emb_module.table_names

            filtered_table_names: List[str] = []
            filtered_table_scores: List[int] = []

            if not set_all_table:
                collection_scores = table_score.get(collection_path, {})
                for name, score in collection_scores.items():
                    if name in table_names:
                        filtered_table_names.append(name)
                        filtered_table_scores.append(score)
            else:
                filtered_table_names = table_names
                filtered_table_scores = [table_score] * len(table_names)

            if not filtered_table_names:
                continue

            dynamic_emb_module.set_score(dict(zip(filtered_table_names, filtered_table_scores)))


def get_score(model: torch.nn.Module) -> Optional[Dict[str, Dict[str, int]]]:
    """Get current score for dynamic embedding tables in the model.

    Args:
        model (torch.nn.Module): Model that contains sharded dynamic embedding
            collections.

    Returns:
        Optional[Dict[str, Dict[str, int]]]: Mapping from collection path to
        table-score mapping.
            - Outer key: embedding collection path in model.
            - Inner key: dynamic embedding table name.
            - Inner value: current score of the table.
        Returns `None` when no sharded dynamic embedding collection or no
        dynamic embedding table is found.
    """
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any ShardedDynamicEmbeddingCollection "
            "or ShardedDynamicEmbeddingBagCollection module, can't get score!",
            UserWarning,
        )
        return None

    check_dynamic_emb_modules_lists: List[List[nn.Module]] = []
    for _, _, tmp_collection_module in collections_list:
        check_dynamic_emb_modules_lists.append(get_dynamic_emb_module(tmp_collection_module))

    has_dynamic_emb = any(len(m_list) > 0 for m_list in check_dynamic_emb_modules_lists)
    if not has_dynamic_emb:
        warnings.warn(
            "Input model don't have any Dynamic embedding tables, can't get score!",
            UserWarning,
        )
        return None

    ret_score_dict: Dict[str, Dict[str, int]] = {}
    for collection_path, _, collection_module in collections_list:
        dynamic_emb_modules = get_dynamic_emb_module(collection_module)

        table_score_map: Dict[str, int] = {}
        for dynamic_emb_module in dynamic_emb_modules:
            # BatchedDynamicEmbeddingTablesV2.get_score 返回 Dict[str, int]
            table_score_map.update(dynamic_emb_module.get_score())
        ret_score_dict[collection_path] = table_score_map
    return ret_score_dict

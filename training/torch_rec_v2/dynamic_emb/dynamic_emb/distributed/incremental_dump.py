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

# pylint: disable=import-error
# pylint: disable=unexpected-keyword-arg
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist

from dynamic_emb.distributed.dump_load import find_sharded_modules, get_dynamic_emb_module
from rec_sdk_common.constants.constants import ValidatorParams
from rec_sdk_common.validator.safe_checker import class_safe_check, int_safe_check


logger = logging.getLogger(__name__)


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


def _validate_incremental_dump_inputs(
    model: nn.Module,
    score_threshold: Union[int, Dict[str, Dict[str, int]]],
    pg: Optional[dist.ProcessGroup],
) -> bool:
    """Validate incremental_dump inputs; return whether to dump all tables."""
    class_safe_check("model", model, (nn.Module,))
    class_safe_check("pg", pg, (dist.ProcessGroup, type(None)))

    if isinstance(score_threshold, int):
        int_safe_check(
            "score_threshold",
            score_threshold,
            min_value=0,
            max_value=ValidatorParams.MAX_INT64.value,
        )
        return True

    if isinstance(score_threshold, dict):
        class_safe_check("score_threshold", score_threshold, (dict,))
        for collection_path, table_thresholds in score_threshold.items():
            class_safe_check("key of score_threshold", collection_path, (str,))
            class_safe_check("value of score_threshold", table_thresholds, (dict,))
            for table_name, threshold in table_thresholds.items():
                class_safe_check("key of table score_threshold", table_name, (str,))
                class_safe_check("threshold of score_threshold", threshold, (int,))
                int_safe_check(
                    "threshold of score_threshold",
                    threshold,
                    min_value=0,
                    max_value=ValidatorParams.MAX_INT64.value,
                )
        return False

    raise ValueError("DynamicEmb Error: score_threshold should be int or Dict[str, Dict[str, int]]")


def set_score(model: torch.nn.Module, table_score: Union[int, Dict[str, Dict[str, int]]]) -> None:
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


def incremental_dump(
    model: torch.nn.Module,
    score_threshold: Union[int, Dict[str, Dict[str, int]]],
    pg: Optional[dist.ProcessGroup] = None,
) -> Union[
    Tuple[
        Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        Dict[str, Dict[str, int]],
    ],
    None,
]:
    """Dump the model's embedding tables incrementally based on the score threshold.
       The index-embedding pair whose score is not less than the threshold will be returned.

    Args:
        model(nn.Module):The model containing dynamic embedding tables.
        score_threshold(Union[int, Dict[str, Dict[str, int]]]):
            int: All embedding table's score threshold will be this integer.
                 It will dump matched results for all tables in the model.
                 Valid range: [0, ValidatorParams.MAX_INT64.value].
            Dict[str, Dict[str, int]]: the first `str` is the name of embedding collection in the model.
                'str' in Dict[str, int] is the name of dynamic embedding table, and `int` in Dict[str, int] is
                the table's score threshold. It will dump for only tables whose names present in this Dict.
                Each int value must be in valid range: [0, ValidatorParams.MAX_INT64.value].
        pg(Optional[dist.ProcessGroup]): optional. The process group used to control the communication scope
            in the dump. Defaults to None.

    Raises:
        ValueError: If `model`, `score_threshold`, or `pg` has an invalid type or value.

    Returns
    -------
    Tuple:
        Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
            The first 'str' is the name of embedding collection.
            The second 'str' is the name of embedding table.
            The first tensor in the Tuple is matched keys on hosts.
            The second tensor in the Tuple is matched values on hosts.
        Dict[str, Dict[str, int]]:
            The first 'str' is the name of embedding collection.
            The second 'str' is the name of embedding table.
            `int` is the current score after finishing the dumping process, which will be used as the score for
                the next forward pass, and can also be used as the input of the next incremental_dump.
                If input score_threshold is `int`, the Dict will contain all dynamic embedding tables' current score,
                otherwise only dumped tables' current score will be returned.
    """

    set_all_table = _validate_incremental_dump_inputs(model, score_threshold, pg)

    logger.debug("[incremental_dump] set_all_table=%s score_threshold=%s", set_all_table, score_threshold)

    # find embedding collections
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any ShardedDynamicEmbeddingCollection module, can't incremental dump!",
            UserWarning,
        )
        return None

    # check if the model have dynamic embedding
    check_dynamic_emb_modules_lists: List[List[nn.Module]] = []

    for tmp_collection in collections_list:
        _, _, tmp_collection_module = tmp_collection
        check_dynamic_emb_modules_lists.append(get_dynamic_emb_module(tmp_collection_module))

    has_dynamic_emb = False
    for check_dynamic_emb_module_list in check_dynamic_emb_modules_lists:
        if len(check_dynamic_emb_module_list) > 0:
            has_dynamic_emb = True
            break

    if not has_dynamic_emb:
        warnings.warn(
            "Input model don't have any Dynamic embedding tables, can't incremental dump!",
            UserWarning,
        )
        return None
    if not set_all_table:
        # filter the embedding collection
        collection_paths_in_module = set()
        filtered_collections_list = []

        for tmp_module_path, tmp_module_name, module in collections_list:
            collection_paths_in_module.add(tmp_module_path)
            if tmp_module_path in score_threshold.keys():
                filtered_collections_list.append((tmp_module_path, tmp_module_name, module))

        collections_list = filtered_collections_list

        # maybe user input shared module name wrong, here raise a warning tell user that model don't have the module name
        for tmp_input_collection_name in score_threshold.keys():
            if tmp_input_collection_name not in collection_paths_in_module:
                warnings.warn(
                    f"'{tmp_input_collection_name}' specified in score_threshold "
                    "not found in the model or not ShardedDynamicEmbeddingCollection instance",
                    UserWarning,
                )

    logger.debug(
        "[incremental_dump] iterating %d collection(s), paths=%s",
        len(collections_list),
        [path for path, _, _ in collections_list],
    )

    ret_tensors: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    ret_scores: Dict[str, Dict[str, int]] = {}
    for tmp_collection in collections_list:
        collection_path, _, tmp_collection_module = tmp_collection
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)

        collection_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        collection_scores: Dict[str, int] = {}

        for dynamic_emb_module in tmp_dynamic_emb_module_list:
            tmp_table_names = dynamic_emb_module.table_names

            filtered_table_names: List[str] = []
            filtered_thresholds: List[int] = []
            if not set_all_table:
                tmp_collection_scores = score_threshold[collection_path]
                tmp_input_names = tmp_collection_scores.keys()
                for name in tmp_input_names:
                    if name in tmp_table_names:
                        index = tmp_table_names.index(name)
                        filtered_table_names.append(tmp_table_names[index])
                        filtered_thresholds.append(tmp_collection_scores[name])
            else:
                filtered_table_names = tmp_table_names
                filtered_thresholds.extend([score_threshold] * len(tmp_table_names))
            if len(filtered_table_names) == 0:
                logger.debug("[incremental_dump] skip collection_path=%s no tables match filter", collection_path)
                continue
            # do incremental dump
            table_thresholds = dict(zip(filtered_table_names, filtered_thresholds))
            logger.debug(
                "[incremental_dump] call BatchedDynamicEmbeddingTablesV2.incremental_dump "
                "collection_path=%s table_thresholds=%s",
                collection_path,
                table_thresholds,
            )
            tensors, scores = dynamic_emb_module.incremental_dump(table_thresholds, pg)
            for tn, (k_t, v_t) in tensors.items():
                logger.debug(
                    "[incremental_dump] got tensors collection_path=%s table=%s "
                    "num_keys=%s num_values=%s score_after=%d",
                    collection_path,
                    tn,
                    k_t.numel(),
                    v_t.numel(),
                    scores.get(tn),
                )
            collection_tensors.update(tensors)
            collection_scores.update(scores)

        ret_tensors[collection_path] = collection_tensors
        ret_scores[collection_path] = collection_scores

    return ret_tensors, ret_scores

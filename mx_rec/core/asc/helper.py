#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

from functools import reduce
from typing import Union, List, Tuple, Dict

import tensorflow as tf

from mx_rec.constants.constants import MAX_INT32
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.merge_table import find_dangling_table, should_skip
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger
from mx_rec.util.normalization import fix_invalid_table_name
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.validator.validator import para_checker_decorator, ValueCompareValidator, ClassValidator, ListValidator, \
    OrValidator, AndValidator


@para_checker_decorator(check_option_list=[
    (["tgt_key_specs", "args_index_list"], ValueCompareValidator, {"target": None},
     ["check_at_least_one_not_equal_to_target"]),
    (["tgt_key_specs", "args_index_list"], ValueCompareValidator, {"target": None},
     ["check_at_least_one_equal_to_target"]),
    ("tgt_key_specs", OrValidator, {"options": [
        (ClassValidator, {"classes": (FeatureSpec, type(None))}),
        (AndValidator, {"options": [
            (ClassValidator, {"classes": (list, tuple)}),
            (ListValidator,
             {
                 "sub_checker": ClassValidator,
                 "list_max_length": MAX_INT32,
                 "list_min_length": 1,
                 "sub_args": {
                     "classes": FeatureSpec
                 }
             },
             ["check_list_length"])
        ]})
    ]}),
    ("args_index_list", OrValidator, {"options": [
        (ClassValidator, {"classes": type(None)}),
        (AndValidator, {"options": [
            (ClassValidator, {"classes": list}),
            (ListValidator, {
                "sub_checker": ClassValidator,
                "list_max_length": MAX_INT32,
                "list_min_length": 1,
                "sub_args": {
                    "classes": int
                }
            },
             ["check_list_length"])
        ]})
    ]}),
    ("table_names", OrValidator, {"options": [
        (ClassValidator, {"classes": type(None)}),
        (AndValidator, {"options": [
            (ClassValidator, {"classes": list}),
            (ListValidator, {
                "sub_checker": ClassValidator,
                "list_max_length": MAX_INT32,
                "list_min_length": 1,
                "sub_args": {
                    "classes": str
                }
            },
             ["check_list_length"])
        ]})
    ]}),
    ("is_training", ClassValidator, {"classes": (bool, type(None))}),
    ("dump_graph", ClassValidator, {"classes": (bool, type(None))}),
])
def get_asc_insert_func(tgt_key_specs=None, args_index_list=None, table_names=None, **kwargs):
    '''
    desperated.
    use create_asc_insert_func_with_specs or create_asc_insert_func_with_agc
    '''
    # condition 1: only tgt_key_specs
    if tgt_key_specs is not None:
        if args_index_list is not None or table_names is not None:
            raise RuntimeError("call get_asc_insert_func in-correctly, when tgt_key_specs is not None, "
                               "please set args_index_list and table_names None.")
        return create_asc_insert_func_with_specs(tgt_key_specs=tgt_key_specs, **kwargs)
    # condition 2: only args_index_list and table_names
    if args_index_list is not None:
        if table_names is None:
            raise RuntimeError("call get_asc_insert_func in-correctly, when args_index_list is not None, "
                               "please set tgt_key_specs None and set table_names correctly.")
        fixed_table_names = [fix_invalid_table_name(table_name) for table_name in table_names]
        return create_asc_insert_func_with_acg(args_index_list=args_index_list,
                                               table_names=fixed_table_names,
                                               **kwargs)
    raise RuntimeError("call get_asc_insert_func in-correctly.")


def create_asc_insert_func_with_specs(tgt_key_specs, **kwargs):
    '''
    feature spec模式
    '''
    return get_asc_insert_func_inner(tgt_key_specs=tgt_key_specs, **kwargs)


@para_checker_decorator(check_option_list=[
    (["args_index_list", "table_names"], ValueCompareValidator, {"target": None},
     ["check_all_not_equal_to_target"]),
])
def create_asc_insert_func_with_acg(args_index_list, table_names, **kwargs):
    '''
    自动改图模式 auto change graph
    '''
    return get_asc_insert_func_inner(args_index_list=args_index_list,
                                     table_names=table_names,
                                     **kwargs)


def get_asc_insert_func_inner(tgt_key_specs=None, args_index_list=None, table_names=None, **kwargs):
    is_training = kwargs.get("is_training", True)
    dump_graph = kwargs.get("dump_graph", False)

    if tgt_key_specs is not None:
        if not isinstance(tgt_key_specs, (list, tuple)):
            tgt_key_specs = [tgt_key_specs]

        def insert_fn_for_feature_specs(*args):
            data_src = args
            if len(args) == 1:
                data_src = args[0]

            read_emb_key_inputs_dict = {
                "insert_tensors": [],
                "table_names": [],
                "feature_spec_names": [],
                "splits": []
            }
            get_target_tensors_with_feature_specs(tgt_key_specs, data_src, is_training, read_emb_key_inputs_dict)
            logger.debug("do_insert with spec for %s", read_emb_key_inputs_dict.get('table_names'))
            return do_insert(args,
                             insert_tensors=read_emb_key_inputs_dict.get("insert_tensors"),
                             splits=read_emb_key_inputs_dict.get("splits"),
                             table_names=read_emb_key_inputs_dict.get("table_names"),
                             input_dict={"is_training": is_training, "dump_graph": dump_graph,
                                         "timestamp": FeatureSpec.use_timestamp(is_training),
                                         "feature_spec_names": read_emb_key_inputs_dict.get("feature_spec_names"),
                                         "auto_change_graph": False})

        insert_fn = insert_fn_for_feature_specs

    else:
        dangling_tables = find_dangling_table(table_names)

        logger.info("In insert found dangling table(s): %s which does not need to be provided to the EmbInfo.",
                    dangling_tables)

        def insert_fn_for_arg_indexes(*args):
            insert_tensors = get_target_tensors_with_args_indexes(args_index_list)

            logger.debug("do_insert without spec for %s", table_names)
            splits = []
            for insert_tensor in insert_tensors:
                split = reduce(lambda x, y: x * y, insert_tensor.shape.as_list())
                splits.append(split if split is not None else tf.math.reduce_prod(tf.shape(insert_tensor)))

            new_insert_tensors, new_splits, new_table_names = [], [], []
            for idx, table_name in enumerate(table_names):
                if table_name in dangling_tables:
                    logger.info("do_insert skip table by graph : %s", table_name)
                    continue

                skip = should_skip(table_name)
                if skip:
                    logger.info("do_insert skip table by keyword: %s", table_name)
                    continue

                new_insert_tensors.append(insert_tensors[idx])
                new_splits.append(splits[idx])
                new_table_names.append(table_names[idx])

            if FeatureSpec.use_timestamp(is_training):
                new_insert_tensors = insert_tensors
                if len(splits) < 1:
                    raise ValueError(f"When use_timestamp is set to True, "
                                     f"the length of the splits list must be greater than or equal to 1.")
                new_splits = splits[1:]

            return do_insert(args,
                             insert_tensors=new_insert_tensors,
                             splits=new_splits,
                             table_names=new_table_names,
                             input_dict={"is_training": is_training, "dump_graph": dump_graph,
                                         "timestamp": FeatureSpec.use_timestamp(is_training),
                                         "feature_spec_names": None,
                                         "auto_change_graph": True})

        insert_fn = insert_fn_for_arg_indexes

    return insert_fn


def merge_feature_id_request(feature_id_list, split_list, table_name_list):
    if not (len(feature_id_list) == len(split_list) and len(split_list) == len(table_name_list)):
        raise RuntimeError(f"shape not match. len(feature_id_list): {len(feature_id_list)},"
                           f"len(split_list): {len(split_list)}"
                           f"len(table_name_list): {len(table_name_list)}")
    feature_id_requests = zip(feature_id_list, split_list, table_name_list)
    if ConfigInitializer.get_instance().modify_graph:
        feature_id_requests = sorted(feature_id_requests, key=lambda x: (x[2]))
    else:
        feature_id_requests = sorted(feature_id_requests, key=lambda x: (x[2], x[0].name))
    logger.debug("features to merge: %s", feature_id_requests)

    last_table_name = None
    last_split = 0
    last_tensorshape_split = 0
    output_feature_id_list = [x[0] for x in feature_id_requests]
    output_split_list = []
    output_tensorshape_split_list = []
    output_table_name_list = []
    for feature_id, split, table_name in feature_id_requests:
        if last_table_name is None or last_table_name == table_name:
            last_table_name = table_name
            last_split += split
            last_tensorshape_split += tf.math.reduce_prod(tf.shape(feature_id))
        else:
            output_table_name_list.append(last_table_name)
            output_split_list.append(last_split)
            output_tensorshape_split_list.append(last_tensorshape_split)
            last_table_name = table_name
            last_split = split
            last_tensorshape_split = tf.math.reduce_prod(tf.shape(feature_id))

    if last_table_name is not None:
        output_table_name_list.append(last_table_name)
        output_split_list.append(last_split)
        output_tensorshape_split_list.append(last_tensorshape_split)
    logger.debug("merge request from %s %s to %s %s", table_name_list, split_list,
                 output_table_name_list, output_split_list)

    list_set = {
        'output_feature_id_list': output_feature_id_list,
        'output_split_list': output_split_list,
        'output_table_name_list': output_table_name_list,
        'output_tensorshape_split_list': output_tensorshape_split_list,
    }
    return list_set


def send_feature_id_request_async(feature_id_list, split_list, table_name_list, input_dict):
    is_training = input_dict["is_training"]
    timestamp = input_dict["timestamp"]
    host_pipeline_ops = import_host_pipeline_ops()
    use_static = ConfigInitializer.get_instance().use_static
    timestamp_feature_id = []

    if timestamp:
        timestamp_feature_id = feature_id_list[:1]
        feature_id_list = feature_id_list[1:]

    list_set = merge_feature_id_request(feature_id_list, split_list, table_name_list)
    feature_id_list = list_set.get("output_feature_id_list")
    split_list = list_set.get("output_split_list")
    table_name_list = list_set.get("output_table_name_list")
    tensorshape_split_list = list_set.get("output_tensorshape_split_list")

    # check training mode order and ensure channel id
    channel_id = ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(
        is_training)

    if timestamp:
        feature_id_list = timestamp_feature_id + feature_id_list
    concat_tensor = tf.concat(feature_id_list, axis=0)

    if len(split_list) == 0 or len(tensorshape_split_list) == 0:
        raise RuntimeError(f"The length of split list can not be 0.")

    if use_static:
        logger.info("read_emb_key_v2(static), table_name_list: %s, split_list: %s", table_name_list, split_list)
        return host_pipeline_ops.read_emb_key_v2(concat_tensor, channel_id=channel_id, splits=split_list,
                                                 emb_name=table_name_list, timestamp=timestamp)

    logger.info("read_emb_key_v2(dynamic), table_name_list: %s, tensorshape_split_list: %s",
                table_name_list, tensorshape_split_list)
    return host_pipeline_ops.read_emb_key_v2_dynamic(concat_tensor, tensorshape_split_list,
                                                     channel_id=channel_id, emb_name=table_name_list,
                                                     timestamp=timestamp)


def do_insert(args, insert_tensors, splits, table_names, input_dict):
    is_training = input_dict["is_training"]
    dump_graph = input_dict["dump_graph"]
    timestamp = input_dict["timestamp"]
    feature_spec_names = input_dict["feature_spec_names"]
    auto_change_graph = input_dict["auto_change_graph"]

    pipeline_op = \
        send_feature_id_request_async(feature_id_list=insert_tensors,
                                      split_list=splits,
                                      table_name_list=table_names,
                                      input_dict={"is_training": is_training,
                                                  "timestamp": timestamp,
                                                  "feature_spec_names": feature_spec_names,
                                                  "auto_change_graph": auto_change_graph})

    if dump_graph:
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        tf.compat.v1.train.write_graph(graph_def, "./export_graph", "pipeline_graph.pb", False)

    # Have to export read_emb_key_v2 op, other wise tensorflow will wipe out it by graph optimizing.
    if auto_change_graph:
        # Args must be: tuple(origin_batch, tuple(read_emb_key_inputs)).
        output_batch = insert_read_emb_key_op_with_modify_graph(args, pipeline_op)
        logger.debug("In do_insert func, the output batch of modify graph is: %s.", output_batch)
        return output_batch
    output_batch = export_read_emb_key_v2_op(args, pipeline_op)
    return output_batch


def insert_read_emb_key_op_with_modify_graph(
        batch: Union[Dict, List, Tuple], pipeline_op: tf.Tensor
) -> Union[Dict, List, Tuple]:
    """
    Insert the read_emb_key operator on the original batch basis.

    Args:
        batch: the batch of read emb key input was inserted into the original batch.
        pipeline_op: the read_emb_key operator.

    Returns: Insert the batch after the operator.

    """

    if not isinstance(batch, tuple) or len(batch) < 2:
        raise ValueError(
            f"The shape must be tuple(origin_batch, tuple(read_emb_key_inputs)), but got: {batch}."
        )

    ori_dataset_spec = ConfigInitializer.get_instance().train_params_config.dataset_element_spec
    if isinstance(ori_dataset_spec, dict):
        original_batch = list(batch)[0]
    else:
        original_batch = list(batch)[:-1]
    if not isinstance(original_batch, (list, tuple, dict)):
        raise TypeError("Dataset batch must be dict/list/tuple type.")

    if isinstance(original_batch, dict):
        insert_key = get_valid_op_key(original_batch)
        original_batch[insert_key] = pipeline_op
    else:
        original_batch.append(pipeline_op)

    if isinstance(ori_dataset_spec, tuple):
        original_batch = tuple(original_batch)

    return original_batch


def export_read_emb_key_v2_op(args, pipeline_op):
    origin_batch = list(args)
    if len(origin_batch) < 1:
        raise ValueError("the length of args is less than 1.")
    if isinstance(origin_batch[0], dict):
        output_batch = origin_batch[0]
        valid_key = get_valid_op_key(output_batch)
        output_batch[valid_key] = pipeline_op

    elif len(origin_batch) == 1 and isinstance(origin_batch[0], tf.Tensor):
        origin_batch.append(pipeline_op)
        output_batch = tuple(origin_batch)

    elif len(origin_batch) == 2:
        if isinstance(origin_batch[0], (list, tuple)):
            origin_batch[0] = list(origin_batch[0])
            origin_batch[0].append(pipeline_op)
            origin_batch[0] = tuple(origin_batch[0])
            output_batch = tuple(origin_batch)

        elif isinstance(origin_batch[0], tf.Tensor):
            origin_batch[0] = [origin_batch[0]]
            origin_batch[0].append(pipeline_op)
            origin_batch[0] = tuple(origin_batch[0])
            output_batch = tuple(origin_batch)

        else:
            raise EnvironmentError(f"An unexpected condition was encountered.")

    else:
        origin_batch.append(tuple(pipeline_op))
        output_batch = tuple(origin_batch)
    return output_batch


def get_valid_op_key(batch_dict: dict) -> str:
    if not isinstance(batch_dict, dict):
        raise TypeError(f"batch_dict must be a dict")

    sorted_keys = sorted(batch_dict)
    valid_key = f"{sorted_keys[-1]}_read_emb_key"

    return valid_key


def get_target_tensors_with_args_indexes(args_index_list):
    insert_tensors = []
    graph = tf.compat.v1.get_default_graph()
    for index in args_index_list:
        tensor = graph.get_tensor_by_name("args_%d:0" % index)
        if tensor.dtype != tf.int64:
            logger.debug("Input tensor dtype is %s, which will be transferred to tf.int64.", tensor.dtype)
            tensor = tf.cast(tensor, tf.int64)
        insert_tensors.append(tf.reshape(tensor, [-1, ]))

    return insert_tensors


def get_target_tensors_with_feature_specs(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict):
    def parse_feature_spec(feature_spec, batch, is_training, read_emb_key_inputs_dict):
        if isinstance(batch, dict):
            if feature_spec.index_key not in batch:
                # feature_spec.is_timestamp is true when batch does not contain timestamp
                if feature_spec.is_timestamp:
                    raise KeyError(f"Cannot find key or index {feature_spec.index_key} in batch.")
                # feature_spec.is_timestamp is false when batch does not contain timestamp
                return

            if not isinstance(batch.get(feature_spec.index_key), tf.Tensor):
                raise TypeError(f"Target value is not a tensor, which is a {type(batch.get(feature_spec.index_key))}.")

            tensor = batch.get(feature_spec.index_key)
        elif isinstance(batch, (list, tuple)):
            if feature_spec.index_key >= len(batch):
                raise ValueError(f"index out of range.")

            if not isinstance(batch[feature_spec.index_key], tf.Tensor):
                raise TypeError(f"Target value is not a tensor, which is a {type(batch[feature_spec.index_key])}.")

            tensor = batch[feature_spec.index_key]
        else:
            raise ValueError(f"Encounter a invalid batch.")

        # Ensure that the sequence of the `read emb key` op input tensor is the same as that of the split result
        # of the multi lookup in a same table.
        reshape_name = "reshape_" + feature_spec.name
        if feature_spec.is_timestamp is None:
            result = feature_spec.set_feat_attribute(tensor, is_training)
            tensor = result.get("tensor")
            table_name = result.get("table_name")
            split = result.get("split")
            if tensor.dtype != tf.int64:
                tensor = tf.cast(tensor, dtype=tf.int64)

            read_emb_key_inputs_dict["insert_tensors"].append(tf.reshape(tensor, [-1, ], name=reshape_name))
            read_emb_key_inputs_dict["table_names"].append(table_name)
            read_emb_key_inputs_dict["splits"].append(split)
            read_emb_key_inputs_dict["feature_spec_names"].append(feature_spec.name)
        elif feature_spec.is_timestamp:
            if len(tensor.shape.as_list()) != 0:
                raise ValueError(f"Given TimeStamp Tensor must be a scalar.")
            read_emb_key_inputs_dict["insert_tensors"] = [tf.reshape(
                tensor, [-1, ], name=reshape_name)] + read_emb_key_inputs_dict.get("insert_tensors", [])
            feature_spec.include_timestamp(is_training)
        elif tensor is not None:
            raise ValueError(f"Spec timestamp should be true when batch contains timestamp.")

    if isinstance(tgt_key_specs, dict):
        for key, item in tgt_key_specs.items():
            get_target_tensors_with_feature_specs(item, batch[key], is_training, read_emb_key_inputs_dict)
        return

    elif isinstance(tgt_key_specs, (list, tuple)):
        if is_feature_spec_list(tgt_key_specs):
            for feature in tgt_key_specs:
                get_target_tensors_with_feature_specs(feature, batch, is_training, read_emb_key_inputs_dict)
            return

        elif isinstance(batch, (list, tuple)) and len(tgt_key_specs) == len(batch):
            for spec, sub_batch in zip(tgt_key_specs, batch):
                get_target_tensors_with_feature_specs(spec, sub_batch, is_training, read_emb_key_inputs_dict)
            return

    elif isinstance(tgt_key_specs, FeatureSpec):
        parse_feature_spec(tgt_key_specs, batch, is_training, read_emb_key_inputs_dict)
        return

    raise ValueError(f"Please keep tgt_key_specs was built with the same structure compare to given batch. \n\t\t"
                     f"In fact, tgt_key_specs type is {type(tgt_key_specs)} but batch type is {type(batch)}.")


def is_feature_spec_list(specs):
    if not isinstance(specs, (list, tuple)):
        return False

    for item in specs:
        if not isinstance(item, FeatureSpec):
            return False

    return True

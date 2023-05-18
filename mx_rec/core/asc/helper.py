#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2021-2023 Huawei Technologies Co., Ltd
import logging
from functools import reduce
import os

import tensorflow as tf

from mx_rec.util.initialize import get_host_pipeline_ops, insert_feature_spec, insert_training_mode_channel_id, \
    get_training_mode_channel_id, get_use_static
from .feature_spec import FeatureSpec


def get_asc_insert_func(tgt_key_specs=None, args_index_list=None, feature_numbers=None,
                        table_names=None, **kwargs):
    '''
    desperated.
    use create_asc_insert_func_with_specs or create_asc_insert_func_with_agc
    '''
    if tgt_key_specs is not None:
        return create_asc_insert_func_with_specs(tgt_key_specs=tgt_key_specs, **kwargs)
    if args_index_list is not None:
        return create_asc_insert_func_with_acg(args_index_list=args_index_list,
                                               feature_counts=feature_numbers,
                                               table_names=table_names,
                                               **kwargs)
    raise RuntimeError("call get_asc_insert_func in-correctly.")


def create_asc_insert_func_with_specs(tgt_key_specs, **kwargs):
    '''
    feature spec模式
    '''
    return get_asc_insert_func_inner(tgt_key_specs=tgt_key_specs, **kwargs)


def create_asc_insert_func_with_acg(args_index_list, feature_counts, table_names, **kwargs):
    '''
    自动改图模式 auto change graph
    '''
    return get_asc_insert_func_inner(args_index_list=args_index_list,
                                     feature_counts=feature_counts,
                                     table_names=table_names,
                                     **kwargs)


def find_dangling_table(table_names):
    def check_tensor(table_name, table_reachable_tensor):
        if ''.join(["/update_", table_name]) in table_reachable_tensor.name \
                or table_reachable_tensor.op.type == 'ApplyAdam':
            return True
        if 'gradients/' in table_reachable_tensor.name:
            return True
        return False

    def find_table_op(table_name, the_op, table_lookup_op, table_reachable_tensor):
        if table_name in the_op.name and the_op.type == "IdentityN":
            if table_name not in table_lookup_op:
                table_lookup_op[table_name] = [the_op]
                table_reachable_tensor[table_name] = the_op.outputs
            else:
                table_lookup_op[table_name].append(the_op)
                table_reachable_tensor[table_name].extend(the_op.outputs)

    op_list = tf.get_default_graph().get_operations()
    table_lookup_op = {}
    table_reachable_tensor = {}
    for the_op in op_list:
        for table_name in table_names:
            find_table_op(table_name, the_op, table_lookup_op, table_reachable_tensor)

    logging.info(f"*********** find tables: {table_lookup_op}***********")
    logging.info(f"looking for dangling table")
    dangling_table = []

    def extend(op_list, tensor, spread_tensors):
        for the_op in op_list:
            if tensor in the_op.inputs:
                spread_tensors.extend(the_op.outputs)

    def bfs_lookup(table_name,next_to_visit):
        tensors_visited = set()
        while next_to_visit:
            spread_tensors = []
            for tensor in next_to_visit:
                if tensor in tensors_visited:
                    continue
                if check_tensor(table_name, tensor):
                    return True
                tensors_visited.add(tensor)
                extend(op_list, tensor, spread_tensors)
            next_to_visit = spread_tensors
        return False

    for table_name, table_op in table_reachable_tensor.items():
        found = bfs_lookup(table_name,table_op)
        if not found:
            dangling_table.append(table_name)
    return dangling_table


def get_asc_insert_func_inner(tgt_key_specs=None, args_index_list=None, feature_counts=None,
                              table_names=None, **kwargs):
    both_none = tgt_key_specs is None and args_index_list is None
    both_no_none = tgt_key_specs is not None and args_index_list is not None
    if both_none or both_no_none:
        raise ValueError("Args tgt_key_specs and args_index_list should and only can choice one to get insert tensors.")

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
                "insert_tensors": [], "table_names": [],
                                        "feature_spec_names": [], "splits": []
            }
            get_target_tensors_with_feature_specs(tgt_key_specs, data_src, is_training, read_emb_key_inputs_dict)
            logging.debug(f"do_insert with spec for {read_emb_key_inputs_dict['table_names']}")
            return do_insert(args,
                             insert_tensors=read_emb_key_inputs_dict["insert_tensors"],
                             splits=read_emb_key_inputs_dict["splits"],
                             table_names=read_emb_key_inputs_dict["table_names"],
                             input_dict={"is_training": is_training, "dump_graph": dump_graph,
                                         "timestamp": FeatureSpec.use_timestamp(is_training),
                                         "feature_spec_names": read_emb_key_inputs_dict["feature_spec_names"],
                                         "auto_change_graph": False})

        insert_fn = insert_fn_for_feature_specs

    else:
        if feature_counts is None or table_names is None:
            raise ValueError("Please config 'args_index_list', 'feature_counts' and 'table_names' at the same time.")
        logging.info(f"all table_names: {table_names}")
        dangling_tables = find_dangling_table(table_names)
        for table_name in dangling_tables:
            logging.info(f"In insert found dangling table: {table_name} "
                         f"which does not need to be provided to the EmbInfo.")
            table_names.remove(table_name)
        logging.info(f"used table_names: {table_names}")

        def insert_fn_for_arg_indexes(*args):
            insert_tensors = get_target_tensors_with_args_indexes(args_index_list)
            # config timestamp later

            logging.debug(f"do_insert without spec for {table_names}")
            splits = []
            for insert_tensor in insert_tensors:
                split = reduce(lambda x, y: x * y, insert_tensor.shape.as_list())
                splits.append(split if split is not None else tf.math.reduce_prod(tf.shape(insert_tensor)))
            return do_insert(args,
                             insert_tensors=insert_tensors,
                             splits=splits,
                             table_names=table_names,
                             input_dict={"is_training": is_training, "dump_graph": dump_graph,
                                         "timestamp": FeatureSpec.use_timestamp(is_training),
                                         "feature_spec_names": None,
                                         "auto_change_graph": True})

        insert_fn = insert_fn_for_arg_indexes

    return insert_fn


def merge_feature_id_request(feature_id_list, split_list, table_name_list, feature_spec_names):
    if not (len(feature_id_list) == len(split_list) and len(split_list) == len(table_name_list)):
        raise RuntimeError(f"shape not match. len(feature_id_list): {len(feature_id_list)},"
                           f"len(split_list): {len(split_list)}"
                           f"len(table_name_list): {len(table_name_list)}")
    feature_id_requests = zip(feature_id_list, split_list, table_name_list)
    feature_id_requests = sorted(feature_id_requests, key=lambda x: (x[2], x[0].name))
    logging.debug(f" features to merge: {feature_id_requests}")
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
    logging.debug(f"merge request from {table_name_list} {split_list} "
                  f" to {output_table_name_list} {output_split_list}")
    return output_feature_id_list, output_split_list, output_table_name_list, output_tensorshape_split_list


def send_feature_id_request_async(feature_id_list, split_list,
                                  table_name_list, input_dict):
    is_training = input_dict["is_training"]
    timestamp = input_dict["timestamp"]
    feature_spec_names = input_dict["feature_spec_names"]
    auto_change_graph = input_dict["auto_change_graph"]
    host_pipeline_ops = get_host_pipeline_ops()
    use_static = get_use_static()
    timestamp_feature_id = []

    if timestamp:
        timestamp_feature_id = feature_id_list[:1]
        feature_id_list = feature_id_list[1:]

    if not auto_change_graph:  # future support acg
        feature_id_list, split_list, table_name_list, tensorshape_split_list = \
            merge_feature_id_request(feature_id_list, split_list,
                                     table_name_list, feature_spec_names)
    else:
        tensorshape_split_list = split_list

    # check training mode order and ensure channel id
    channel_id = get_training_mode_channel_id(is_training=is_training)

    if timestamp:
        feature_id_list = timestamp_feature_id + feature_id_list
    concat_tensor = tf.concat(feature_id_list, axis=0)

    if use_static:
        logging.debug(f"read_emb_key_v2(static), table_name_list: {table_name_list}, split_list: {split_list}")
        return host_pipeline_ops.read_emb_key_v2(concat_tensor, channel_id=channel_id, splits=split_list,
                                                 emb_name=table_name_list, timestamp=timestamp)

    logging.debug(f"read_emb_key_v2_dynamic, table_name_list: {table_name_list}, "
                  f"tensorshape_split_list: {tensorshape_split_list}")
    pipeline_op = host_pipeline_ops.read_emb_key_v2_dynamic(concat_tensor, tensorshape_split_list,
                                                            channel_id=channel_id,
                                                            emb_name=table_name_list, timestamp=timestamp)

    return pipeline_op


def do_insert(args, insert_tensors, splits, table_names, input_dict):
    is_training = input_dict["is_training"]
    dump_graph = input_dict["dump_graph"]
    timestamp = input_dict["timestamp"]
    feature_spec_names = input_dict["feature_spec_names"]
    auto_change_graph = input_dict["auto_change_graph"]

    # Only the tables that need to be used after table combination are retained in meituan situation.
    # Current solution has error in same situations. For example, a sparse table has not been auto-merged.
    from mx_rec.util.constants import ASCEND_TABLE_NAME_MUST_CONTAIN
    new_insert_tensors, new_splits, new_table_names = [], [], []
    logging.debug(f"In do_insert function, ASCEND_TABLE_NAME_MUST_CONTAIN: {ASCEND_TABLE_NAME_MUST_CONTAIN}")
    for idx, table_name in enumerate(table_names):
        if ASCEND_TABLE_NAME_MUST_CONTAIN is not None and ASCEND_TABLE_NAME_MUST_CONTAIN not in table_name:
            logging.info(f"After the tables are combined, the information about the"
                         f" {table_name} table does not need to be provided to the read_emb_key operator.")
            continue
        new_insert_tensors.append(insert_tensors[idx])
        new_splits.append(splits[idx])
        new_table_names.append(table_names[idx])

    if timestamp:
        new_insert_tensors = insert_tensors

    pipeline_op = \
        send_feature_id_request_async(feature_id_list=new_insert_tensors,
                                      split_list=new_splits,
                                      table_name_list=new_table_names,
                                      input_dict={"is_training": is_training,
                                                  "timestamp": timestamp,
                                                  "feature_spec_names": feature_spec_names,
                                                  "auto_change_graph": auto_change_graph})

    if dump_graph:
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        tf.compat.v1.train.write_graph(graph_def, "./export_graph", "pipeline_graph.pb", False)

    # have to export read_emb_key_v2 op, other wise tensorflow will wipe out it by graph optimizing
    output_batch = export_read_emb_key_v2_op(args, pipeline_op)
    return output_batch


def export_read_emb_key_v2_op(args, pipeline_op):
    origin_batch = list(args)
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
    graph = tf.get_default_graph()

    for index in args_index_list:
        tensor = graph.get_tensor_by_name("args_%d:0" % index)
        if tensor.dtype != tf.int64:
            logging.debug(f"Input tensor dtype is {tensor.dtype}, which will be transferred to tf.int64.")
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

        if feature_spec.is_timestamp is None:
            tensor, table_name, feat_count, split = feature_spec.set_feat_attribute(tensor, is_training)
            if tensor.dtype != tf.int64:
                tensor = tf.cast(tensor, dtype=tf.int64)

            read_emb_key_inputs_dict["insert_tensors"].append(tf.reshape(tensor, [-1, ]))
            read_emb_key_inputs_dict["table_names"].append(table_name)
            read_emb_key_inputs_dict["splits"].append(split)
            read_emb_key_inputs_dict["feature_spec_names"].append(feature_spec.name)
        elif feature_spec.is_timestamp:
            if len(tensor.shape.as_list()) != 0:
                raise ValueError(f"Given TimeStamp Tensor must be a scalar.")
            read_emb_key_inputs_dict["insert_tensors"] = [tf.reshape(tensor, [-1, ])] + \
                                                         read_emb_key_inputs_dict["insert_tensors"]
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


def get_asc_read_raw_func(cfg_list):
    batch = {}
    int_name_order = []
    int_len_list = []
    float_name_order = []
    float_len_list = []
    line_per_sample_list = []
    host_pipeline_ops = get_host_pipeline_ops()
    for cfg in cfg_list:
        if cfg.data_type == "int64":
            int_name_order.append(cfg.feature_name)
            int_len_list.append(cfg.feature_len)
            line_per_sample_list.append(cfg.line_per_sample)

        if cfg.data_type == "float":
            float_name_order.append(cfg.feature_name)
            float_len_list.append(cfg.feature_len)
            line_per_sample_list.append(cfg.line_per_sample)
    if len(set(line_per_sample_list)) != 1:
        raise ValueError(f"Please check that each line_per_sample value should be equal.")
    line_per_sample = line_per_sample_list[0]

    def read_raw_fn(data_src):
        raw_int_sample, raw_float_sample = host_pipeline_ops.read_raw(
            sample=data_src,
            int_len=sum(int_len_list) * line_per_sample,
            float_len=sum(float_len_list) * line_per_sample,
            feat_order=int_name_order + float_name_order
        )

        int_split_res = tf.split(raw_int_sample, [i * line_per_sample_list[0] for i in int_len_list])

        float_split_res = tf.split(raw_float_sample, [i * line_per_sample_list[0] for i in float_len_list])

        logging.debug(f"############ Enter read_raw_fn ########")

        for name_id, name in enumerate(int_name_order):
            batch[name] = int_split_res[name_id]

        for name_id, name in enumerate(float_name_order):
            batch[name] = float_split_res[name_id]
        return batch

    return read_raw_fn


class ParseConfig:

    def __init__(self, **kwargs):
        self.input_keys = set(kwargs.keys())
        self._feature_name = kwargs.get("feature_name")
        self._feature_len = int(kwargs.get("feature_len"))
        self._data_type = kwargs.get("data_type")
        self._line_per_sample = int(kwargs.get("line_per_sample"))
        self.check_params()

    @property
    def feature_name(self):
        return self._feature_name

    @property
    def feature_len(self):
        return self._feature_len

    @property
    def data_type(self):
        return self._data_type

    @property
    def line_per_sample(self):
        return self._line_per_sample

    def check_params(self):
        supported_keys = {"feature_name", "feature_len", "line_per_sample", "data_type"}
        if self.input_keys != supported_keys:
            raise KeyError("Please offer an expected keyword argument")

        if not isinstance(self._feature_name, str):
            raise TypeError(f"Please offer a feature_name with string type.")

        if not isinstance(self._data_type, str):
            raise TypeError(f"Please offer a data_type with string type.")

        if self._data_type not in ("int64", "float"):
            raise TypeError(f"Please offer a data_type with int64 or float type")

        if self._feature_len <= 0:
            raise ValueError(f"Please offer a feature_len greater than zero.")

        if self._line_per_sample <= 0:
            raise ValueError(f"Please offer a line_per_sample greater than zero.")

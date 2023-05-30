#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc
from mx_rec.core.asc.helper import get_asc_insert_func
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import SparseEmbedding
from mx_rec.util.constants import ASCEND_CUTTING_POINT_INITIALIZER, ASCEND_SPARSE_LOOKUP_ENTRANCE, \
    ASCAnchorAttr, ASCEND_TIMESTAMP
from mx_rec.util.initialize import get_rank_size, get_training_mode_channel_id, get_feature_spec, \
    insert_feature_spec, set_initializer, get_use_static, get_use_hot, get_device_id, get_use_dynamic_expansion, \
    terminate_config_initializer
from mx_rec.util.perf import performance
from mx_rec.graph.utils import check_input_list, find_parent_op, check_cutting_points, replace_anchor, \
    record_ops_to_replace, export_pb_graph, make_sorted_key_to_tensor_list


def get_preprocessing_map_func(graph_def, input_names, output_names, batch_tensor_names=None,
                               pipeline_input_indexes=None):
    input_names = check_input_list(input_names, str)
    output_names = check_input_list(output_names, str)
    batch_tensor_names = check_input_list(batch_tensor_names, str)
    pipeline_input_indexes = check_input_list(pipeline_input_indexes, int)
    both_is_none = batch_tensor_names is None and pipeline_input_indexes is None
    both_not_none = batch_tensor_names is not None and pipeline_input_indexes is not None
    if both_is_none or both_not_none:
        raise ValueError("It is legal when and only when one of the parameters 'batch_tensor_names' and "
                         "'pipeline_input_indexes' was given.")

    def map_func(*args):
        def print_tensors(batch_id, tracker=None):
            if tracker is None:
                tracker = []
            if isinstance(batch_id, dict):
                for key, item in batch_id.items():
                    print_tensors(item, tracker + [key])
            if isinstance(batch_id, tf.Tensor):
                logging.debug(f"######## tracker: {tracker}, tensor: {batch_id} ########")

        for batch in args:
            print_tensors(batch)

        batch = args[0]

        input_tensors = []
        if batch_tensor_names is not None:
            for tensor_name in batch_tensor_names:
                tensor = batch.get(tensor_name)
                if tensor is None:
                    raise ValueError(f"Given input_tensor_name '{tensor_name}' is invalid.")

                input_tensors.append(tensor)

        else:
            graph = tf.get_default_graph()
            for index in pipeline_input_indexes:
                tensor = graph.get_tensor_by_name("args_%d:0" % index)
                input_tensors.append(tensor)

        output_list = tf.import_graph_def(graph_def, input_map=dict(zip(input_names, input_tensors)),
                                          return_elements=output_names)

        output_batch = list(args)
        output_batch.append(tuple(output_list))
        return tuple(output_batch)

    return map_func


def get_input_index_list(cutting_point_list, replacement_specs, mapping_name_list, base_count, timestamp_index=None):
    input_index_list = []
    for cutting_point in cutting_point_list:
        if cutting_point in replacement_specs:
            index = int(cutting_point.name.split(":")[1])

        elif cutting_point.name in mapping_name_list:
            index = base_count + mapping_name_list.index(cutting_point.name)

        else:
            raise ValueError(f"Cannot find a matching output for cutting point tensor named '{cutting_point.name}'.")
        input_index_list.append(index)
    if timestamp_index is not None:
        input_index_list = [timestamp_index] + input_index_list

    return input_index_list


def find_make_iterator_op(batch_tensor):
    graph = tf.get_default_graph()
    operations = graph.get_operations()
    for each_op in operations:
        for input_tensor in batch_tensor.op.inputs:
            if input_tensor.op.outputs and input_tensor.op.outputs[0] in list(
                    each_op.inputs) and each_op.type == "MakeIterator":
                logging.debug(f"Op MakeIterator '{each_op.name}' was found.")
                return each_op

    raise ValueError(f"Op MakeIterator was not found.")


@performance("find_target_dataset_op")
def find_target_dataset_op(base_ops, op_type):
    base_ops = check_input_list(base_ops, tf.Operation)
    parent_ops = base_ops

    while True:
        for parent_op in parent_ops:
            if parent_op.type == op_type:
                return parent_op

        base_ops = parent_ops
        parent_ops = []
        for base_op in base_ops:
            parent_ops.extend(find_parent_op(base_op))

        if not parent_ops:
            raise ValueError(f"Op {op_type} was not found.")


def get_op_before_optimize_dataset(get_next_op):
    if get_next_op.type != "IteratorGetNext":
        raise TypeError("Op '{get_next_op}' must be one instance of IteratorGetNext.")

    # looking for the MakeIterator operator which corresponds to given batch_tensor
    base_op = find_make_iterator_op(get_next_op.outputs[0])
    # looking for the op which is the one before OptimizeDataset operator
    target_op = find_target_dataset_op(base_op, "OptimizeDataset")
    if find_parent_op(target_op)[0].type == "PrefetchDataset":
        target_op = find_parent_op(target_op)[0]

    return target_op


def get_passing_tensor_list(src_tensors, target_op):
    def get_passing_tensors(src_tensor):
        passing_tensors = []
        tensor_list = [src_tensor]
        while tensor_list:
            last_tensor = tensor_list.pop()
            if last_tensor.op is target_op:
                passing_tensors.append(last_tensor)
            else:
                tensor_list.extend(list(last_tensor.op.inputs))

        return passing_tensors

    src_tensors = check_input_list(src_tensors, tf.Tensor)
    passing_tensor_list = []
    sub_src_tensors = []
    for tensor in src_tensors:
        passing_tensors = get_passing_tensors(tensor)
        for passing_tensor in passing_tensors:
            if passing_tensor not in passing_tensor_list:
                passing_tensor_list.append(passing_tensor)
        if len(passing_tensors) != 0:
            logging.info(f"passing_tensors: {passing_tensors}")
            sub_src_tensors.append(tensor)
        else:
            logging.info(f"Cannot find passing tensor for given tensor '{tensor}'.")

    output_index_list = [int(tensor.name.split(":")[1]) for tensor in passing_tensor_list]

    return passing_tensor_list, output_index_list, sub_src_tensors


def find_target_instance_dataset(variant_tensor):
    dataset_instance_list = tf.compat.v1.get_collection("dataset_group")
    for ins in dataset_instance_list:
        if ins._variant_tensor == variant_tensor:
            if not isinstance(ins, DatasetV1Adapter):
                ins = ins._input_dataset
            logging.debug(f"Find target instance '{ins}', whose variant_tensor is '{variant_tensor}'.")
            if not isinstance(ins.element_spec, dict) and not (
                    isinstance(ins.element_spec, (list, tuple)) and len(ins.element_spec) == 2 and isinstance(
                ins.element_spec[0], dict)):
                raise NotImplementedError("The found dataset does not return a valid layout.")

            return ins

    raise LookupError(f"Can not find target instance, whose variant_tensor is '{variant_tensor}' respectively.")


def get_sub_graph(input_tensors, output_tensors):
    input_tensors = check_input_list(input_tensors, tf.Tensor)
    output_tensors = check_input_list(output_tensors, tf.Tensor)
    input_op_name_list = [tensor.op.name for tensor in input_tensors]
    output_op_name_list = [tensor.op.name for tensor in output_tensors]

    graph_def = tf.compat.v1.get_default_graph().as_graph_def()
    cut_graph_input = tf.compat.v1.graph_util.extract_sub_graph(graph_def, input_op_name_list)
    cut_graph_output = tf.compat.v1.graph_util.extract_sub_graph(graph_def, output_op_name_list)

    node_list = []
    node_list_input = cut_graph_input.node
    node_list_output = cut_graph_output.node
    for node in node_list_output:
        if node not in node_list_input:
            node_list.append(node)

    sub_graph_def = tf.compat.v1.GraphDef()
    sub_graph_def.node.extend(node_list)

    input_name_list = [tensor.name for tensor in input_tensors]
    output_name_list = [tensor.name for tensor in output_tensors]

    return sub_graph_def, input_name_list, output_name_list


def update_input_tensor_with_new_batch(replacement_specs, new_get_next_op_name):
    graph = tf.compat.v1.get_default_graph()
    for old_tensor, item in replacement_specs.items():
        for idx, operator in item:
            old_tensor_name = old_tensor.name
            output_index = old_tensor_name.split(":")[-1]
            new_tensor_name = "%s:%s" % (new_get_next_op_name, output_index)
            new_tensor = graph.get_tensor_by_name(new_tensor_name)
            operator._update_input(idx, new_tensor)


def make_src_to_tgt_mapping(src_element_spec, tgt_element_spec):
    # adding '_0' to the prefix
    if not isinstance(src_element_spec, (list, tuple)):
        src_element_spec = [src_element_spec]
    src_sorted_keys = make_sorted_key_to_tensor_list(src_element_spec, [])
    tgt_sorted_keys = make_sorted_key_to_tensor_list(tgt_element_spec, [])
    index_to_src_key_mapping = dict([(idx, key) for idx, key in enumerate(src_sorted_keys)])
    tgt_key_to_index_mapping = dict([(key, idx) for idx, key in enumerate(tgt_sorted_keys)])

    original_tensor_count = len(src_sorted_keys)

    def mapping_func(src_idx):
        key = index_to_src_key_mapping.get(src_idx)
        if key is None:
            raise ValueError("Given src_idx is out of range.")

        tgt_idx = tgt_key_to_index_mapping.get(key)
        return tgt_idx

    return mapping_func, original_tensor_count


@performance("graph_modifier")
def modify_graph_and_start_emb_cache(dump_graph=False):
    modify_graph_for_asc(dump_graph=dump_graph)
    start_asc_pipeline()


def generate_get_next_op_specs(cutting_point_list, dump_graph):
    get_next_op_map = defaultdict(dict)
    for input_tensor in cutting_point_list:
        get_next_op = find_target_dataset_op(input_tensor.op, "IteratorGetNext")
        if get_next_op not in get_next_op_map:
            logging.debug(f"find a new get_next_op named '{get_next_op.name}'")
            replacement_specs = record_ops_to_replace(get_next_op)
            get_next_op_map[get_next_op]["replacement_specs"] = replacement_specs
            passing_tensor_list, batch_tensor_index_list, sub_cutting_point_list = \
                get_passing_tensor_list(cutting_point_list, get_next_op)
            get_next_op_map[get_next_op]["passing_tensor_list"] = passing_tensor_list
            get_next_op_map[get_next_op]["batch_tensor_index_list"] = batch_tensor_index_list
            get_next_op_map[get_next_op]["sub_cutting_point_list"] = sub_cutting_point_list

            sub_graph_def, input_name_list, output_name_list = get_sub_graph(passing_tensor_list,
                                                                             sub_cutting_point_list)
            get_next_op_map[get_next_op]["sub_graph_def"] = sub_graph_def
            get_next_op_map[get_next_op]["input_name_list"] = input_name_list
            get_next_op_map[get_next_op]["output_name_list"] = output_name_list
            get_next_op_map[get_next_op]["is_training"] = \
                SparseEmbedding.get_anchor_attribute(input_tensor, ASCAnchorAttr.IS_TRAINING)

            export_pb_graph("cut_graph_%s.pb" % get_next_op.name, dump_graph, graph_def=sub_graph_def)

    return get_next_op_map


def get_src_and_generate_tgt_dataset(get_next_op, records):
    target_op = get_op_before_optimize_dataset(get_next_op)
    src_dataset = find_target_instance_dataset(target_op.outputs[0])
    tgt_dataset = src_dataset.map(get_preprocessing_map_func(records.get("sub_graph_def"),
                                                             records.get("input_name_list"),
                                                             records.get("output_name_list"),
                                                             pipeline_input_indexes=records.get(
                                                                 "batch_tensor_index_list")))

    return src_dataset, tgt_dataset


def modify_graph_for_asc(dump_graph=False, prefetch=10):
    cutting_point_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE)
    check_cutting_points(cutting_point_list)
    if not cutting_point_list:
        logging.warning("Nothing to revise.")
        return

    export_pb_graph("old_graph.pb", dump_graph)
    get_next_op_map = generate_get_next_op_specs(cutting_point_list, dump_graph)

    for get_next_op, records in get_next_op_map.items():
        is_training = records.get("is_training")
        timestamp_index = get_timestamp_index(get_next_op, is_training)
        src_dataset, tgt_dataset = get_src_and_generate_tgt_dataset(get_next_op, records)
        mapping_func, original_tensor_count = make_src_to_tgt_mapping(src_dataset.element_spec,
                                                                      tgt_dataset.element_spec)
        sub_cutting_point_list = records.get("sub_cutting_point_list")
        input_index_list = get_input_index_list(sub_cutting_point_list,
                                                records.get("replacement_specs"),
                                                records.get("output_name_list"),
                                                original_tensor_count, timestamp_index=timestamp_index)
        feature_numbers = [SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC).feat_cnt for
                           cutting_point in sub_cutting_point_list]
        table_names = [SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC).table_name for
                       cutting_point in sub_cutting_point_list]
        tgt_dataset = tgt_dataset.map(
            get_asc_insert_func(feature_numbers=feature_numbers, table_names=table_names,
                                args_index_list=input_index_list, is_training=is_training, dump_graph=dump_graph))

        tgt_dataset = tgt_dataset.prefetch(prefetch)
        new_iterator = tgt_dataset.make_initializable_iterator()
        new_batch = new_iterator.get_next()
        tf.compat.v1.add_to_collection(ASCEND_CUTTING_POINT_INITIALIZER, new_iterator.initializer)
        set_initializer(is_training, new_iterator.initializer)

        try:
            one_tensor = [v for _, v in new_batch.items()][0]
        except IndexError as err:
            raise IndexError(f"Cannot find a tensor from given batch.") from err
        new_get_next_op_name = find_target_dataset_op(one_tensor.op, "IteratorGetNext").name
        update_input_tensor_with_new_batch(records.get("replacement_specs"), new_get_next_op_name)

        for _, cutting_point in enumerate(sub_cutting_point_list):
            feature_spec = SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC)
            table_instance = SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.TABLE_INSTANCE)
            channel_id = get_training_mode_channel_id(is_training)
            config = dict(
                batch_size=feature_spec.batch_size, feat_cnt=feature_spec.feat_cnt,
                send_count=table_instance.send_count, channel_id=channel_id, rank_size=get_rank_size(),
                table_name=table_instance.table_name, skip_emb_transfer=table_instance.skip_emb_transfer,
                ext_emb_size=table_instance.ext_emb_size, emb_size=table_instance.scalar_emb_size,
                use_hot=get_use_hot(), device_id=get_device_id(), use_dynamic_expansion=get_use_dynamic_expansion())
            build_asc_graph(table_instance, cutting_point, config, is_training)

    logging.info("Graph has been revised.")
    export_pb_graph("new_graph.pb", dump_graph)


def get_timestamp_index(get_next_op, is_training):
    timestamp_tensor_list = tf.compat.v1.get_collection(ASCEND_TIMESTAMP)
    timestamp_index = None
    for timestamp in timestamp_tensor_list:
        if timestamp in get_next_op.outputs:
            timestamp_index = int(timestamp.name.split(":")[1])
            timestamp_feature_spec = get_feature_spec("timestamp")
            if timestamp_feature_spec is None:
                timestamp_feature_spec = FeatureSpec("timestamp", index_key=timestamp_index, is_timestamp=True)
                timestamp_feature_spec.include_timestamp(is_training)
                insert_feature_spec(timestamp_feature_spec, is_training)
                break

            if timestamp_feature_spec.index_key != timestamp_index:
                raise ValueError(f"Given timestamp_index, which is {timestamp_index}, does not match index "
                                 f"key. Please double check.")
            timestamp_feature_spec.include_timestamp(is_training)
            break
    return timestamp_index


def build_asc_graph(table_instance, cutting_point, config, is_training):
    # returned results swap_pos and swap_len were not in used, will be applied for DDR mode
    logging.debug(f"try to replace anchors for table {config.get('table_name')} on channel {config.get('channel_id')}")
    skip_emb_transfer = config.get("skip_emb_transfer")
    logging.info(f"modifier build_asc_graph skip_emb_transfer: {skip_emb_transfer}")

    if len(table_instance.channel_name_list) > 1:
        channel_name_queue = table_instance.channel_name_dict.get(is_training)
        if len(channel_name_queue) < 1:
            raise ValueError(f"The length of channel_name_queue must be greater than or equal to 1.")
        ids_channel_name = channel_name_queue.pop()
        config["send_count"] = table_instance.send_count_map.get(ids_channel_name)
    elif len(table_instance.channel_name_list) == 1:
        ids_channel_name = config.get('table_name')
    else:
        raise ValueError(f"The length of channel_name_list must be greater than or equal to 1.")

    if skip_emb_transfer:
        result = get_preprocessed_tensor_for_asc(table_instance.variable, config, ids_channel_name,
                                                 table_instance.modify_graph)
    else:
        variable_list = [table_instance.variable] \
                        + [slot_info.get("slot") for slot_info in table_instance.optimizer_slot_info_list]
        result = get_preprocessed_tensor_for_asc(variable_list, config, ids_channel_name, table_instance.modify_graph)
    restore_vector = result.get("restore_vector")
    hot_pos = result.get("hot_pos")
    id_offsets = result.get("id_offsets")
    swap_in = result.get("swap_in")
    all2all_matrix = result.get("all2all_args")

    with tf.control_dependencies(swap_in):
        id_offsets = tf.identity(id_offsets)

    logging.info(f"build_asc_graph -> id_offsets: {id_offsets}")
    replace_anchor_vec(cutting_point, ASCAnchorAttr.ID_OFFSETS, id_offsets)
    logging.info(f"build_asc_graph -> restore_vector: {restore_vector}")
    replace_anchor_vec(cutting_point, ASCAnchorAttr.RESTORE_VECTOR, restore_vector)

    logging.info(f"build_asc_graph -> all2all_matrix: {all2all_matrix}")
    if not get_use_static():
        replace_anchor_vec(cutting_point, ASCAnchorAttr.ALL2ALL_MATRIX, all2all_matrix)

    logging.info(f"build_asc_graph -> hot_pos: {hot_pos}")
    if get_use_hot():
        replace_anchor_vec(cutting_point, ASCAnchorAttr.HOT_POS, hot_pos)

    logging.debug(f"has replace anchors for table {config.get('table_name')} on channel {config.get('channel_id')}")


def replace_anchor_vec(cutting_point, attribute, anchor):
    anchor_vec = SparseEmbedding.get_anchor_attribute(cutting_point, attribute)
    replacement_specs_for_anchor_vec = record_ops_to_replace(anchor_vec.op)
    replace_anchor(replacement_specs_for_anchor_vec, [anchor])


class GraphModifierHook(tf.estimator.SessionRunHook):
    def __init__(self, dump_graph=True, modify_graph=False):
        self.dump_graph = dump_graph
        self.modify_graph = modify_graph

    def begin(self):
        if self.modify_graph:
            modify_graph_and_start_emb_cache(dump_graph=self.dump_graph)
        else:
            start_asc_pipeline()

    def after_create_session(self, session, coord):
        if self.modify_graph:
            session.run(tf.compat.v1.get_collection(ASCEND_CUTTING_POINT_INITIALIZER))

    def end(self, session):
        terminate_config_initializer()

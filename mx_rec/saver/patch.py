#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
# Some code is derived from Tensorflow, which is subject to the following copyright notice:
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# We pick up the code of Tensorflow to make the api of mxRec compatible with Tensorflow for model saving and loading.
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

import os
import time
import logging
import json

import tensorflow as tf
from tensorflow.compat.v1.summary import FileWriter
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import graph_io
from tensorflow.python.ops import variables
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training_util
from tensorflow.python.util import compat
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow_estimator.python.estimator.hooks.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.core.util.event_pb2 import SessionLog
import numpy as np
from mpi4py import MPI

from mx_rec.saver.saver import Saver as SparseSaver
from mx_rec.saver.saver import check_file_system_is_valid, should_write_data, update_model_index, \
    write_delta_export_time_ms, get_model_type_by_version, get_base_and_delta_models, read_base_delta_and_write
from mx_rec.util.communication.hccl_ops import get_rank_id
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.validator.validator import para_checker_decorator, ClassValidator, StringValidator, OptionalIntValidator, \
    OptionalStringValidator, DirectoryValidator
from mx_rec.util.log import logger
from mx_rec.constants.constants import MAX_INT32, INVALID_CHARS

_FILENAME_SUFFIX = "filename_suffix"


def get_sparse_vars(var_list):
    sparse_var_list = []
    # build sparse saver
    if var_list is not None:
        if not isinstance(var_list, (list, tuple)):
            raise TypeError("A non-None var_list must be a list or tuple.")
        ascend_variables = tf.compat.v1.get_collection(
            ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection)
        for var in var_list:
            if var in ascend_variables:
                sparse_var_list.append(var)
    else:
        sparse_var_list = tf.compat.v1.get_collection(
            ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection)
    return sparse_var_list


def init_check(defer_build, var_list):
    if defer_build and var_list:
        raise ValueError(
            "If `var_list` is provided then build cannot be deferred. Either set defer_build=False or var_list=None.")
    if context.executing_eagerly():
        tf_logging.warning("When executing eagerly variables do not necessarily have unique names, "
                           "and so the variable.name-based lookups Saver performs are error-prone.")
        if var_list is None:
            raise RuntimeError("eager execution, `var_list` must specify a list or dict of variables to save")


def saver_init(self, var_list=None, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0,
               name=None, restore_sequentially=False, saver_def=None, builder=None, defer_build=False,
               allow_empty=False, write_version=saver_pb2.SaverDef.V2, pad_step_number=False, save_relative_paths=False,
               filename=None, fid_version=0):
    self._var_list = var_list
    self._last_checkpoints = []
    self._checkpoints_to_be_deleted = []
    self._is_built = False
    self._is_empty = None
    init_check(defer_build, var_list)
    self._write_version = write_version
    self._reshape = reshape
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self._save_relative_paths = save_relative_paths
    self._sharded = sharded
    self._restore_sequentially = restore_sequentially
    self._max_to_keep = max_to_keep
    self._builder = builder
    self._name = name
    self._filename = filename
    self.saver_def = saver_def
    self._allow_empty = allow_empty
    self._pad_step_number = pad_step_number
    # mt customed parameter
    self._fid_version = fid_version

    # mxRec Patch
    # create sparse saver only when sparse_var_list is not None
    self.sparse_saver = None
    sparse_var_list = get_sparse_vars(var_list)
    if sparse_var_list:
        self.sparse_saver = SparseSaver(var_list=sparse_var_list, max_to_keep=max_to_keep, prefix_name=filename)

    if self.saver_def:
        self._check_saver_def()
        self._write_version = self.saver_def.version

    if context.executing_eagerly():
        keep_time = self._keep_checkpoint_every_n_hours * 3600
        self._next_checkpoint_time = (time.time() + keep_time)
    elif not defer_build:
        self.build()
    self._object_restllore_saver = None


def save_check(latest_filename, sess):
    if os.path.split(latest_filename)[0]:
        raise ValueError("'latest_filename' must not contain path components")
    if not context.executing_eagerly() and not isinstance(sess, session.SessionInterface):
        raise TypeError(f"'sess' must be a Session; {sess}")


def get_model_checkpoint_path(self, checkpoint_file, sess):
    if not context.executing_eagerly():
        model_checkpoint_path = sess.run(self.saver_def.save_tensor_name,
                                         {self.saver_def.filename_tensor_name: checkpoint_file})
        logger.info("Save dense model into dir %s", checkpoint_file)
    else:
        self._build_eager(checkpoint_file, build_save=True, build_restore=False)
        model_checkpoint_path = self.saver_def.save_tensor_name

    return model_checkpoint_path


def update_checkpoint_state(self, model_checkpoint_path, parent_save_path, latest_file_name, suffix_meta_graph,
                            save_path):
    self._RecordLastCheckpoint(model_checkpoint_path)
    try:
        checkpoint_management.update_checkpoint_state_internal(save_dir=parent_save_path,
                                                               model_checkpoint_path=model_checkpoint_path,
                                                               all_model_checkpoint_paths=self.last_checkpoints,
                                                               latest_filename=latest_file_name,
                                                               save_relative_paths=self._save_relative_paths)
    except errors.NotFoundError as err:
        if not gfile.IsDirectory(parent_save_path):
            err = ValueError(f"Parent directory of {save_path} doesn't exist, can't save.")
        raise err
    self._MaybeDeleteOldCheckpoints(meta_graph_suffix=suffix_meta_graph)


def write_meta_graph_task(self, **kwargs):
    checkpoint_file = kwargs.get("checkpoint_file")
    meta_graph_suffix = kwargs.get("meta_graph_suffix")
    sess = kwargs.get("sess")
    strip_default_attrs = kwargs.get("strip_default_attrs")
    save_debug_info = kwargs.get("save_debug_info")

    meta_graph_name = checkpoint_management.meta_graph_filename(checkpoint_file, meta_graph_suffix=meta_graph_suffix)
    if not context.executing_eagerly():
        with sess.graph.as_default():
            self.export_meta_graph(meta_graph_name, strip_default_attrs=strip_default_attrs,
                                   save_debug_info=save_debug_info)


def get_checkpoint_file(self, global_step, sess, save_path):
    if not isinstance(global_step, compat.integral_types):
        global_step = training_util.global_step(sess, global_step)
    checkpoint_file = f"{save_path}-{global_step}"
    if self._pad_step_number:
        checkpoint_file = f"{save_path}-{global_step:08d}"
    return checkpoint_file


def build(self):
    self._var_list = build_var_list()
    if context.executing_eagerly():
        raise RuntimeError("Use save/restore instead of build in eager mode.")
    self._build(self._filename, build_save=True, build_restore=True)


def check_characters_is_valid(characters: str) -> bool:
    if any(c in INVALID_CHARS for c in characters):
        return False

    return True


@para_checker_decorator(check_option_list=[
    ("sess", ClassValidator, {"classes": (tf.compat.v1.Session, tf.compat.v1.train.MonitoredSession)}),
    ("save_path", StringValidator, {"min_len": 1, "max_len": 150}, ["check_string_length"]),
    ("global_step", ClassValidator, {"classes": (int, np.int64, type(None))}),
    ("global_step", OptionalIntValidator, {"min_value": 0, "max_value": MAX_INT32}, ["check_value"]),
    ("latest_filename", ClassValidator, {"classes": (str, type(None))}),
    ("latest_filename", OptionalStringValidator, {"min_len": 1, "max_len": 50}, ["check_string_length"]),
    ("meta_graph_suffix", ClassValidator, {"classes": (str, type(None))}),
    ("meta_graph_suffix", OptionalStringValidator, {"min_len": 1, "max_len": 50}, ["check_string_length"]),
    ("write_meta_graph", ClassValidator, {"classes": (bool, type(None))}),
    ("write_state", ClassValidator, {"classes": (bool, type(None))}),
    ("strip_default_attrs", ClassValidator, {"classes": (bool, type(None))}),
    ("save_debug_info", ClassValidator, {"classes": (bool, type(None))})
])
def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix="meta", write_meta_graph=True,
         write_state=True, strip_default_attrs=False, save_debug_info=False, is_incremental_checkpoint=False,
         save_delta=False):
    msg = "Saving model by normal pattern."
    if is_incremental_checkpoint:
        start_save_time = time.time()
        saved_model_type = "delta" if save_delta else "base"
        msg = f"Saving {saved_model_type} model by incremental checkpoint pattern."
    tf_logging.info(msg)
    if not check_characters_is_valid(save_path):
        raise ValueError("save_path contains invalid characters such as newline, formfeed,"
                         " carriage return, backspace, tab, vertical tab, and delete.")

    if not check_file_system_is_valid(save_path):
        raise ValueError("the path to save belong to invalid file system, only local file system supported. ")

    if not self._is_built and not context.executing_eagerly():
        raise RuntimeError("`build()` should be called before save if defer_build==True")

    if latest_filename is None:
        latest_filename = "checkpoint"

    if self._write_version != saver_pb2.SaverDef.V2:
        tf_logging.warning("TensorFlow's V1 checkpoint format has been deprecated.")

    save_check(latest_filename, sess)

    if global_step is not None:
        checkpoint_file = get_checkpoint_file(self, global_step, sess, save_path)
    else:
        checkpoint_file = save_path
        if os.path.basename(save_path) == latest_filename and not self._sharded:
            # Guard against collision between data file and checkpoint state file.
            raise ValueError(f"{latest_filename} collides with {save_path}")

    save_path_parent = os.path.dirname(save_path)
    model_checkpoint_path = None
    if self._is_empty:
        return model_checkpoint_path

    # mxRec Patch
    # save sparse model, only run when self.sparse_saver is not None
    if not context.executing_eagerly() and self.sparse_saver:
        self.sparse_saver.save(sess, save_path=checkpoint_file, save_delta=save_delta)
        logger.info("Save sparse model into dir %s", checkpoint_file)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Barrier()
    if should_write_data(rank, save_path):
        model_checkpoint_path = compat.as_str(get_model_checkpoint_path(self, checkpoint_file, sess))
        if write_state:
            update_checkpoint_state(self, model_checkpoint_path, save_path_parent, latest_filename, meta_graph_suffix,
                                    save_path)
        if write_meta_graph:
            write_meta_graph_task(self, checkpoint_file=checkpoint_file, meta_graph_suffix=meta_graph_suffix, sess=sess,
                                  strip_default_attrs=strip_default_attrs, save_debug_info=save_debug_info)
    comm.Barrier()
    if is_incremental_checkpoint:
        save_cost_time = time.time() - start_save_time
        save_dir, _ = os.path.split(save_path)
        export_tag = "Seconds" if save_delta else "DueTime"
        model_index_info = {"timestamp": str(int(start_save_time)), "export_tag": export_tag, "type": saved_model_type,
                            "global_step": int(global_step), "cost_ms": int(save_cost_time * 1000)}
        if save_delta:
            delta_model_version = "delta_" + str(int(global_step))
            write_delta_export_time_ms(save_dir, {delta_model_version: int(save_cost_time * 1000)})
        update_model_index(save_dir, model_index_info)

    # TODO: 当保存base的时候清空delta目录
    # if not save_delta:
    #     clear_delta_model()
    return model_checkpoint_path


@para_checker_decorator(check_option_list=[
    ("sess", ClassValidator, {"classes": (tf.compat.v1.Session, tf.compat.v1.train.MonitoredSession)}),
    ("save_path", StringValidator, {"min_len": 1, "max_len": 150}, ["check_string_length"]),
])
def restore(self, sess, save_path):
    if save_path is None:
        raise ValueError("Can't load save_path when it is None.")

    # 加载场景
    # 不开增量保存加载：和原来的加载逻辑一致，直接加载最新的model即可
    # 开启增量保存加载：
    # 1、指定版本：加载对应版本的model，获取对应model的类型
    # 2、不指定版本：直接加载最新的model，获取对应model的类型
    # 3、当拿到需要加载的模型类型时，需要分情况：
    # 全量（base）model：直接去加载对应目录下的模型即可
    # 增量（delta）model：先找到对应最近的全量（base）model，读取其数据；然后找到这个全量（base）model之后的所有增量（delta）model，依次
    # 加载增量（delta）model，即读取其中的数据并将其更新的加载起来的全量（base）model中，然后创建一个临时目录，用于保存加载起来的model。

    is_incremental_checkpoint = ConfigInitializer.get_instance().is_incremental_checkpoint
    restore_model_version = ConfigInitializer.get_instance().restore_model_version
    is_first_restore = ConfigInitializer.get_instance().is_first_restore
    model_type = "base"

    directory, base_name = os.path.split(save_path)
    if is_incremental_checkpoint:
        if not restore_model_version:
            # 开启增量保存加载且不指定版本：加载最新的model
            restore_model_version = base_name.split("-")[1]
            model_type = get_model_type_by_version(directory, restore_model_version)
            if model_type == "delta":
                # 加载delta
                # 获取离此delta model最近的一个base model，以及这个base model和此delta model之间所有的delta models（包括这个delta
                # model）
                base_model, delta_models = get_base_and_delta_models(directory, restore_model_version)
                logger.info(f"Restore {model_type} model from base model: {base_model} and delta models: "
                            f"{','.join(delta_models)}.")
                read_base_delta_and_write(directory, base_model, delta_models)
            save_path = os.path.join(directory, base_name)
        else:
            base_name = base_name.split("-")[0] + "-" + restore_model_version
            model_type = get_model_type_by_version(directory, restore_model_version)
            if model_type == "delta":
                # 加载delta
                # 获取离此delta model最近的一个base model，以及这个base model和此delta model之间所有的delta models（包括这个delta
                # model）
                base_model, delta_models = get_base_and_delta_models(directory, restore_model_version)
                delta_models_str = " ".join(delta_models)
                logger.info(f"Restore {model_type} model from base model: {base_model} and delta models: "
                            f"{delta_models_str}.")
                read_base_delta_and_write(directory, base_model, delta_models)
            save_path = os.path.join(directory, base_name)
    else:
        model_type = "base"
        save_path = os.path.join(directory, base_name)

    if not check_characters_is_valid(save_path):
        raise ValueError("save_path contains invalid characters such as newline, "
                         "formfeed, carriage return, backspace, tab, vertical tab, and delete.")

    if not check_file_system_is_valid(save_path):
        raise ValueError(f"the path to restore belong to invalid file system, only local file system supported. ")

    if save_path.find("://") == -1:
        directory_validator = DirectoryValidator("reading_path", save_path)
        directory_validator.check_not_soft_link()
        directory_validator.with_blacklist(exact_compare=False)
        directory_validator.check()

    checkpoint_prefix = compat.as_text(save_path)
    if self._is_empty:
        return
    if not checkpoint_management.checkpoint_exists_internal(checkpoint_prefix):
        raise ValueError("the passed save_path is not a valid checkpoint: " +
                         checkpoint_prefix)

    tf_logging.info("Restoring parameters from %s", checkpoint_prefix)
    try:
        if not context.executing_eagerly():
            # mxRec Patch
            # restore sparse model, only run when self.sparse_saver is not None
            if self.sparse_saver:
                self.sparse_saver.restore(sess, save_path, model_type=model_type)

            sess.run(self.saver_def.restore_op_name,
                     {self.saver_def.filename_tensor_name: save_path})
            logger.info("Restore from dir %s", save_path)
        else:
            self._build_eager(save_path, build_save=False, build_restore=True)

    except errors.NotFoundError as err:
        try:
            names_to_keys = object_graph_key_mapping(save_path)
        except errors.NotFoundError:
            raise _wrap_restore_error_with_msg(
                err, "a Variable name or other graph key that is missing") from err

        # This is an object-based checkpoint. We'll print a warning and then do
        # the restore.
        tf_logging.warning(
            "Restoring an object-based checkpoint using a name-based saver. This "
            "may be somewhat fragile, and will re-build the Saver. Instead, "
            "consider loading object-based checkpoints using tf.train.Checkpoint().")
        self._object_restore_saver = saver_from_object_based_checkpoint(checkpoint_path=save_path,
                                                                        var_list=self._var_list, builder=self._builder,
                                                                        names_to_keys=names_to_keys,
                                                                        cached_saver=self._object_restore_saver)

    except errors.InvalidArgumentError as err:
        raise _wrap_restore_error_with_msg(err, "a mismatch between the current graph and the graph") from err


def object_graph_key_mapping(file_path):
    reader = pywrap_tensorflow.NewCheckpointReader(file_path)
    obj_graph_str = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
    obj_graph_proto = (trackable_object_graph_pb2.TrackableObjectGraph())
    obj_graph_proto.ParseFromString(obj_graph_str)
    node_name_to_key = {}
    for each_node in obj_graph_proto.nodes:
        for attribute in each_node.attributes:
            node_name_to_key[attribute.full_name] = attribute.checkpoint_key
    return node_name_to_key


def _wrap_restore_error_with_msg(err, extra_verbiage):
    err_msg = ("Restoring from checkpoint failed."
               "This is most likely due to {} from the checkpoint."
               "Please ensure that you have not altered the graph expected based on the checkpoint. "
               "Original error: {}").format(extra_verbiage, err.message)
    return err.__class__(err.node_def, err.op, err_msg)


def saver_from_object_based_checkpoint(checkpoint_path, var_list=None, builder=None, names_to_keys=None,
                                       cached_saver=None):
    if names_to_keys is None:
        try:
            names_to_keys = object_graph_key_mapping(checkpoint_path)
        except errors.NotFoundError as err:
            raise ValueError(f"Checkpoint in {checkpoint_path} not an object-based checkpoint.") from err
    if var_list is None:
        var_list = build_var_list()

    if builder is None:
        builder = BulkSaverBuilder()

    current_node_names = set()
    obj_saveable_list = saveable_object_util.validate_and_slice_inputs(var_list)

    for obj_saveable in obj_saveable_list:
        for spec in obj_saveable.specs:
            current_node_names.add(spec.name)
    previous_node_names = set(names_to_keys.keys())
    missing_names = current_node_names - previous_node_names
    if missing_names:
        extra_node_names = previous_node_names - current_node_names
        intersecting_names = previous_node_names.intersection(current_node_names)
        raise errors.NotFoundError(
            None, None,
            message=("Existing variables not in the checkpoint: %s\n"
                     "Variables names when this checkpoint was written which don't exist now: %s\n\n"
                     "(%d variable name(s) did match)\n\n"
                     "Could not find some variables in the checkpoint (see names above). "
                     "Saver was attempting to load an object-based checkpoint (saved using tf.train.Checkpoint "
                     "or tf.keras.Model.save_weights) using variable names. "
                     "If the checkpoint was written with eager execution enabled, "
                     "it's possible that variable names have changed (for example missing a '_1' suffix). "
                     "It's also possible that there are new variables which did not exist "
                     "when the checkpoint was written. "
                     "You can construct a Saver(var_list=...) with only the variables which previously existed, "
                     "and if variable names have changed you may need to make this a dictionary "
                     "with the old names as keys. If you're using an Estimator, "
                     "you'll need to return a tf.train.Saver inside a tf.train.Scaffold from your model_fn.") % (
                        ", ".join(sorted(missing_names)), ", ".join(sorted(extra_node_names)), len(intersecting_names)))
    for obj_saveable in obj_saveable_list:
        for spec in obj_saveable.specs:
            spec.name = names_to_keys.get(spec.name)
    if cached_saver is None:
        return tf.compat.v1.train.Saver(obj_saveable_list)
    return cached_saver


def build_var_list():
    save_var_list = []
    tmp_list = variables._all_saveable_objects()
    removing_var_list = ConfigInitializer.get_instance().sparse_embed_config.removing_var_list
    for var in tmp_list:
        if var.name not in removing_var_list:
            save_var_list.append(var)
    return save_var_list


class BaseSaverBuilder(object):
    VariableSaveable = saveable_object_util.ReferenceVariableSaveable
    SaveSpec = saveable_object.SaveSpec
    ResourceVariableSaveable = saveable_object_util.ResourceVariableSaveable
    SaveableObject = saveable_object.SaveableObject

    def __init__(self, write_version=saver_pb2.SaverDef.V2):
        self._write_version = write_version

    def save_op(self, file_name, obj_saveable_list):
        tensors, tensor_names, tensor_slices = [], [], []
        for obj_saveable in obj_saveable_list:
            for spec in obj_saveable.specs:
                tensors.append(spec.tensor)
                tensor_names.append(spec.name)
                tensor_slices.append(spec.slice_spec)
        if self._write_version == saver_pb2.SaverDef.V2:
            return io_ops.save_v2(file_name, tensor_names, tensor_slices,
                                  tensors)
        elif self._write_version == saver_pb2.SaverDef.V1:
            return io_ops._save(filename=file_name, tensor_names=tensor_names, tensors=tensors,
                                tensor_slices=tensor_slices)
        else:
            raise RuntimeError("Unexpected write_version: " + self._write_version)


class BulkSaverBuilder(BaseSaverBuilder):
    def bulk_restore(self, filename_tensor, saveables, preferred_shard, restore_sequentially):
        restore_specs = []
        del restore_sequentially
        for obj_saveable in saveables:
            for spec in obj_saveable.specs:
                restore_specs.append((spec.name,
                                      spec.slice_spec,
                                      spec.dtype))
        tensor_names, tensor_slices, tensor_dtypes = zip(*restore_specs)
        with ops.device("cpu:0"):
            return io_ops.restore_v2(filename_tensor, tensor_names, tensor_slices, tensor_dtypes)


def patch_for_write_graph_func(func):
    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # In the case of multiple processes, choose one process to write graph.
        if len(args) > 1 and should_write_data(rank, args[1]):
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def patch_for_saver():
    dense_saver = tf.compat.v1.train.Saver
    dense_saver.__init__ = saver_init
    dense_saver.save = save
    dense_saver.restore = restore
    dense_saver.build = build
    logger.debug("Class tf.train.Saver has been patched.")
    training_util.write_graph = patch_for_write_graph_func(graph_io.write_graph)


def _patch_for_summary_writer(func):
    def wrapper(*args, **kwargs):
        filename_suffix = kwargs.get(_FILENAME_SUFFIX, "")
        filename_suffix = filename_suffix or ""
        rank_suffix = "_rank" + str(get_rank_id())
        if rank_suffix not in filename_suffix:
            filename_suffix = rank_suffix + "_" + filename_suffix if filename_suffix else rank_suffix
        kwargs[_FILENAME_SUFFIX] = filename_suffix
        return func(*args, **kwargs)

    return wrapper


def patch_for_summary_writer():
    """
    Patch for `tf.summary.FileWriter.__init__` method, add rankId to init param `filename_suffix`.
    """
    FileWriter.__init__ = _patch_for_summary_writer(FileWriter.__init__)
    logger.debug("Method `tf.summary.FileWriter.__init__` has been patched.")


def second_or_step_timer_init(self, every_secs=None, every_steps=None, is_incremental_checkpoint=False):
    self.reset()
    self._every_secs = every_secs
    self._every_steps = every_steps

    from mx_rec.util.initialize import ConfigInitializer
    self._save_checkpoint_due_time = ConfigInitializer.get_instance().save_checkpoint_due_time
    self._save_delta_checkpoints_secs = ConfigInitializer.get_instance().save_delta_checkpoints_secs
    self._is_incremental_checkpoint = is_incremental_checkpoint
    self._is_delta = False
    self._last_triggered_base_time = None
    self._last_triggered_delta_time = None
    self._is_first_update = True

    if self._every_secs is None and self._every_steps is None:
        raise ValueError("Either every_secs or every_steps should be provided.")
    if (self._every_secs is not None) and (self._every_steps is not None):
        raise ValueError("Can not provide both every_secs and every_steps.")
    if is_incremental_checkpoint:
        if (self._save_checkpoint_due_time is None) or (self._save_delta_checkpoints_secs is None):
            raise ValueError("Both save_checkpoint_due_time and save_delta_checkpoints_secs should be provided.")

    super(SecondOrStepTimer, self).__init__()


def should_trigger_for_step(self, step):
    if self._last_triggered_step is None:
        return True

    if self._last_triggered_step == step:
        return False

    if not self._is_incremental_checkpoint:
        if self._every_secs is not None:
            if time.time() >= self._last_triggered_time + self._every_secs:
                return True

        if self._every_steps is not None:
            if step >= self._last_triggered_step + self._every_steps:
                return True
        return False

    if self._save_checkpoint_due_time is not None:
        if time.time() >= self._last_triggered_base_time + self._save_checkpoint_due_time:
            self._is_delta = False
            return True

    if self._save_delta_checkpoints_secs is not None:
        if time.time() >= self._last_triggered_delta_time + self._save_delta_checkpoints_secs:
            self._is_delta = True
            return True

    return False


def update_last_triggered_step(self, step):
    current_time = time.time()
    if self._last_triggered_time is None:
        elapsed_secs = None
        elapsed_steps = None
    else:
        elapsed_secs = current_time - self._last_triggered_time
        elapsed_steps = step - self._last_triggered_step

    self._last_triggered_time = current_time

    if self._is_incremental_checkpoint and self._is_first_update:
        self._last_triggered_base_time = current_time
        self._last_triggered_delta_time = current_time
        self._last_triggered_step = step
        self._is_first_update = False
        return (elapsed_secs, elapsed_steps)

    if self._is_incremental_checkpoint:
        if self._is_delta:
            self._last_triggered_delta_time = current_time
        else:
            self._last_triggered_base_time = current_time

    self._last_triggered_step = step
    return (elapsed_secs, elapsed_steps)


def patch_for_second_or_step_timer():
    second_or_step_timer = tf.compat.v1.train.SecondOrStepTimer
    second_or_step_timer.__init__ = second_or_step_timer_init
    second_or_step_timer.should_trigger_for_step = should_trigger_for_step
    second_or_step_timer.update_last_triggered_step = update_last_triggered_step
    logger.info("Class 'tf.compat.v1.train.SecondOrStepTimer' has been patched.")


def checkpoint_saver_hook_init(self, checkpoint_dir, save_secs=None, save_steps=None, saver=None,
                               checkpoint_basename="model.ckpt", scaffold=None, listeners=None):
    logging.info("Create CheckpointSaverHook.")
    if saver is not None and scaffold is not None:
        raise ValueError("You cannot provide both saver and scaffold.")
    self._saver = saver
    self._checkpoint_dir = checkpoint_dir
    self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
    self._scaffold = scaffold
    from mx_rec.util.initialize import ConfigInitializer
    self._timer = SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)
    self._is_incremental_checkpoint = ConfigInitializer.get_instance().is_incremental_checkpoint
    if self._is_incremental_checkpoint:
        self._timer = SecondOrStepTimer(every_secs=save_secs, every_steps=save_steps,
                                        is_incremental_checkpoint=self._is_incremental_checkpoint)
    self._listeners = listeners or []
    self._steps_per_run = 1


def after_run_checkpoint_saver_hook(self, run_context, run_values):
    stale_global_step = run_values.results
    if self._timer.should_trigger_for_step(stale_global_step +
                                           self._steps_per_run):
        # get the real value after train op.
        global_step = run_context.session.run(self._global_step_tensor)
        if self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            if self._save(run_context.session, global_step, self._timer._is_delta):
                run_context.request_stop()


def save_checkpoint_saver_hook(self, session, step, save_delta=False):
    logging.info("Saving checkpoints for %d into %s.", step, self._save_path)

    for listener in self._listeners:
        listener.before_save(session, step)
    # 新增参数save_delta表示保存的是base还是delta，is_incremental_checkpoint
    self._get_saver().save(session, self._save_path, global_step=step,
                           is_incremental_checkpoint=self._timer._is_incremental_checkpoint,
                           save_delta=save_delta)
    self._summary_writer.add_session_log(
        SessionLog(
            status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
        step)

    should_stop = False
    for listener in self._listeners:
        if listener.after_save(session, step):
            logging.info(
                "A CheckpointSaverListener requested that training be stopped. "
                "listener: {}".format(listener))
            should_stop = True
    return should_stop


def patch_for_checkpoint_saver_hook():
    checkpoint_saver_hook = tf.compat.v1.train.CheckpointSaverHook
    checkpoint_saver_hook.__init__ = checkpoint_saver_hook_init
    checkpoint_saver_hook._save = save_checkpoint_saver_hook
    checkpoint_saver_hook.after_run = after_run_checkpoint_saver_hook
    logger.info("Class 'tf.compat.v1.train.CheckpointSaverHook' has been patched.")

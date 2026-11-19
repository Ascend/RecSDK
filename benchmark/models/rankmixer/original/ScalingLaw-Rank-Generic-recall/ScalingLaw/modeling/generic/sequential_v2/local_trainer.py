import functools
import time
import logging
from typing import Callable, Iterable, Optional, Any, Tuple

import os
import torch
import torch.multiprocessing as mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import queue

from modeling.generic.sequential_v2.GR_model import GR_model
from modeling.generic.sequential_v2.features import seq_features_from_row_cpu, SequentialFeatures


NPU_ENABLE = True
if os.environ.get("NPU_FLAG", "True") == "False":
    NPU_ENABLE = False

if NPU_ENABLE:
    import torch_npu

ASYNC_LOADER_FUNC = None


def assemble_model_executable_callback(
    model: GR_model,
    optimizer_cb: Optional[Callable[[Iterable[torch.Tensor]], optim.Optimizer]],
    local_rank,
    device,
    train_conf,
    feature_conf,
    find_unused_parameters=False
):
    # Define async function
    def sync_data_loader(data_loader, data_queue, include_candidate_items=False, is_recall=False):
        gr_output_length = train_conf.get("gr_output_length", 0)
        batch_limit = train_conf.get("batch_limit", 0)
        for idx, row in enumerate(iter(data_loader)):
            if batch_limit and idx > batch_limit:
                break
            seq_feature_names = [k for k, _ in feature_conf['item_feature_columns'].items()]
            if not include_candidate_items:
                data = seq_features_from_row_cpu(
                    row, max_output_length=gr_output_length,
                    seq_item_feature_names=seq_feature_names,
                    user_feature_names=[k for k, _ in feature_conf['user_feature_columns'].items()],
                    include_loss_weights=True, itemid_column_name=feature_conf['infer_items_key'],
                    infer_ratings_key=feature_conf["infer_ratings_key"],
                    infer_timestamps_key=feature_conf["infer_timestamps_key"]
                )
            else:
                data = seq_features_from_row_cpu(
                    row, max_output_length=gr_output_length + 1 if not is_recall else gr_output_length,
                    seq_item_feature_names=seq_feature_names,
                    user_feature_names=[k for k, _ in feature_conf['user_feature_columns'].items()],
                    include_loss_weights=True, itemid_column_name=feature_conf['infer_items_key'],
                    include_candidate_items=include_candidate_items,
                    infer_ratings_key=feature_conf["infer_ratings_key"],
                    infer_timestamps_key=feature_conf["infer_timestamps_key"]
                )
            data_queue.put(data)
            break
        return
    def async_data_loader(data_loader, data_queue, include_candidate_items=False, is_recall=False):
        gr_output_length = train_conf.get("gr_output_length", 0)
        batch_limit = train_conf.get("batch_limit", 0)
        for idx, row in enumerate(iter(data_loader)):
            if batch_limit and idx > batch_limit:
                break
            seq_feature_names = [k for k, _ in feature_conf['item_feature_columns'].items()]
            if not include_candidate_items:
                data = seq_features_from_row_cpu(
                    row, max_output_length=gr_output_length,
                    seq_item_feature_names=seq_feature_names,
                    user_feature_names=[k for k, _ in feature_conf['user_feature_columns'].items()],
                    include_loss_weights=True, itemid_column_name=feature_conf['infer_items_key'],
                    infer_ratings_key=feature_conf["infer_ratings_key"],
                    infer_timestamps_key=feature_conf["infer_timestamps_key"]
                )
            else:
                data = seq_features_from_row_cpu(
                    row, max_output_length=gr_output_length + 1 if not is_recall else gr_output_length,
                    seq_item_feature_names=seq_feature_names,
                    user_feature_names=[k for k, _ in feature_conf['user_feature_columns'].items()],
                    include_loss_weights=True, itemid_column_name=feature_conf['infer_items_key'],
                    include_candidate_items=include_candidate_items,
                    infer_ratings_key=feature_conf["infer_ratings_key"],
                    infer_timestamps_key=feature_conf["infer_timestamps_key"]
                )
            data_queue.put(data)

        data_queue.put("end")
        # Sleep to ensure the result of `queue.empty()` absolutely accurate
        time.sleep(1e-100)
        while not data_queue.empty():
            time.sleep(1)
        return
    
    # Set global func
    global ASYNC_LOADER_FUNC
    ASYNC_LOADER_FUNC = async_data_loader
    global SYNC_LOADER_FUNC
    SYNC_LOADER_FUNC = sync_data_loader
    # 4. 初始化优化器

    # 6. 训练模型
    model = model.to(device)
    
    opt = None
    if optimizer_cb is not None:
        opt = optimizer_cb(model.parameters())

    if NPU_ENABLE:
        h2d_stream = torch_npu.npu.Stream(device)
    else:
        h2d_stream = torch.cuda.Stream(device)
    use_loss_weighted_grad = train_conf.get("use_loss_weighted_grad", False)

    only_one_batch = {
        "only_one_batch_enable": False,
        "load_already": False,
        "cached_data":None
    }
    ENABLE_SYNC_LOADER = int(os.environ.get('ENABLE_SYNC_LOADER', 0))
    if ENABLE_SYNC_LOADER == 1:
        only_one_batch["only_one_batch_enable"] = True
        
    def exec_cb(data_queue, is_recall, iter=0):
        if only_one_batch['only_one_batch_enable']:
            if not only_one_batch['load_already']:
                queue_data = data_queue.get()
                only_one_batch['load_already'] = True
                only_one_batch['cached_data'] = queue_data
            else:
                queue_data = only_one_batch['cached_data']
        else:
            queue_data = data_queue.get()
        if queue_data == "end":
            raise StopIteration
        payloads_tod = {}
        if NPU_ENABLE:
            with torch_npu.npu.stream(h2d_stream):
                uid_device = queue_data.uid.to(device, non_blocking=True)
                historical_lengths_device = queue_data.past_lengths.to(device, non_blocking=True)
                for name, feature in queue_data.past_payloads.items():
                    payloads_tod[name] = feature.to(device, non_blocking=True)
        else:
            with torch.cuda.stream(h2d_stream):
                uid_device = queue_data.uid.to(device, non_blocking=True)
                historical_lengths_device = queue_data.past_lengths.to(device, non_blocking=True)
                for name, feature in queue_data.past_payloads.items():
                    payloads_tod[name] = feature.to(device, non_blocking=True)

            
        seq_features = SequentialFeatures(
            uid=uid_device,
            past_lengths=historical_lengths_device,
            past_ids=payloads_tod.get(feature_conf.get('infer_items_key')),
            past_payloads=payloads_tod,
        )

        if model.training:
            opt.zero_grad()

        model_input = seq_features.past_payloads
        model_input['past_lengths'] = seq_features.past_lengths
        model_input['past_ids'] = seq_features.past_ids
        model_input.pop("user_id", None)

        output = model(model_inputs=model_input) if model.training or not is_recall else None
        if model.training:
            opt.step()
        return output, seq_features

    return model, exec_cb


def feed_datas_to_exec_callback(
    exec_cb: Callable[[Any, int], Tuple[torch.Tensor, SequentialFeatures]],
    is_recall: bool,
    train_data_loader,
    eval_data_loader
) -> Tuple[Optional[Callable[[int], Tuple[torch.Tensor, SequentialFeatures]]], Optional[
    Callable[[int], Tuple[torch.Tensor, SequentialFeatures]]]]:
    global ASYNC_LOADER_FUNC
    if ASYNC_LOADER_FUNC is None:
        raise RuntimeError("'feed_datas_to_exec_callback' must be called after 'assemble_model_executable_callback'")

    train_exec_cb = None 
    eval_exec_cb = None
    global SYNC_LOADER_FUNC
    if train_data_loader is not None:
        train_data_queue = mp.Queue(maxsize=3)
        train_loader_proc = mp.Process(target=ASYNC_LOADER_FUNC, args=(train_data_loader, train_data_queue))
        train_loader_proc.start()
        train_exec_cb = functools.partial(exec_cb, train_data_queue, is_recall)
    ENABLE_SYNC_LOADER = int(os.environ.get('ENABLE_SYNC_LOADER', 0))
    if eval_data_loader is not None:
        if ENABLE_SYNC_LOADER == 1:
            logging.info("enable sync loader.")
            eval_data_queue = queue.Queue()
            SYNC_LOADER_FUNC(eval_data_loader, eval_data_queue, True, is_recall)
        else:
            logging.info("enable async loader.")
            eval_data_queue = mp.Queue(maxsize=3)
            eval_loader_proc = mp.Process(target=ASYNC_LOADER_FUNC, args=(eval_data_loader, eval_data_queue, True, is_recall))
            eval_loader_proc.start()
        eval_exec_cb = functools.partial(exec_cb, eval_data_queue, is_recall)
    return train_exec_cb, eval_exec_cb

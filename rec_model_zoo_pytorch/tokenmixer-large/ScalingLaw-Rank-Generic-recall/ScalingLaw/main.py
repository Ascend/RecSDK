import copy
import json
import logging
import os
import pickle
import random
import sys
import time
from math import sqrt
from statistics import mean
from typing import Any, Callable, Dict, Tuple, Iterable, Union
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from absl import app, flags
from sklearn.metrics import roc_auc_score
from torch import optim
from modeling.model_registry import ModelRegistry
from modeling.model_initializer import ModelInitializer
from const_global import data_target
from data.data_loader import create_data_loader
from data.eval import process_rank_scores, gather_all_list, avg_eval, MetricsCalculator
from data.reco_dataset import get_reco_dataset
from modeling.generic.sequential_v2.encoder_utils import get_sequential_encoder_v2
from modeling.generic.sequential_v2.features import SequentialFeatures
from modeling.generic.sequential_v2.local_trainer import assemble_model_executable_callback, \
    feed_datas_to_exec_callback
from modeling.generic.utils.constants import Const
from utils.common_utils import get_config, refine_feat_and_model_conf
from data.eval_recall import eval_metrics_v2_from_tensors
from data.eval_recall import get_eval_state
from data.eval_recall import _avg
from modeling.generic.indexing.mips_top_k import get_top_k_module
from torch.utils.tensorboard import SummaryWriter
import datetime
import collections
from utils.common_utils import get_unique_path, to_clean_cpu_tensor
logger = logging.getLogger(__name__)
NPU_ENABLE = True
if os.environ.get("NPU_FLAG", "True") == "False":
    NPU_ENABLE = False
    print("model run on GPU!!!")
if NPU_ENABLE:
    import torch_npu
    from torch_npu.profiler import profile
# 初始化传入参数
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
flags.DEFINE_string("config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
flags.DEFINE_string("data_dir", None, "Path to data.")
flags.DEFINE_string("save_dir", None, "Path to save.")
flags.DEFINE_string("feature_map_dir", None, "Path of feature_map.")
flags.DEFINE_string("feature_map_max_index_path", None, "Path of feature map max index")
flags.DEFINE_string("period", None, "Period of task execution.")
flags.DEFINE_string("tensorboard_log_dir", None, "Period of save log.")
flags.DEFINE_boolean("is_train", True, "If the model is training or testing.")
flags.DEFINE_boolean("save_user_emb", True, "If the model is used for saving user emb")
flags.DEFINE_boolean("get_infer_result", False, "If save the input and output data for precision alignment.")
flags.DEFINE_integer("eval_batch_size",None,"Override eval_batch_size in config file")
FLAGS = flags.FLAGS

def init_ddp_info_params(already_init=False):
    """
    初始化加速器参数
    """
    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_num = int(os.environ.get("NNODES", 1))
    logging.info(f"tcp://{addr}:{port}, nnodes={node_num}, rank={rank_id}")
    # initialize the process group
    if not already_init:
        # initialize the process group
        if NPU_ENABLE:
            dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)
        else:
            dist.init_process_group("nccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)
    if NPU_ENABLE:
        return f"npu:{local_rank}", rank_id, local_rank, world_size, node_num
    else:
        return f"cuda:{local_rank}", rank_id, local_rank, world_size, node_num

def init_random_seed(config):
    """
    设置全局随机数以使训练结果更确定
    """
    random_seed = config.get('seed_conf', {}).get("global_seed", '1234')
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if NPU_ENABLE:
        torch_npu.npu.manual_seed(random_seed)
        torch_npu.npu.manual_seed_all(random_seed)

def init_torch_config(local_rank, train_conf):
    """
    设置torch配置
    """
    use_tf32 = train_conf.get("use_tf32", False)
    if NPU_ENABLE:
        torch_npu.npu.set_device(local_rank)
        torch_npu.npu.matmul.allow_hf32=True
    else:
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    logging.info("cuda.matmul.allow_tf32: %s", use_tf32)
    logging.info("cudnn.allow_tf32: %s", use_tf32)

def get_data_loaders(dataset, data_dir, rank, world_size, train_config, model_conf, feature_config, dataloader_config):
    """
    获取训练和验证数据 dataloader
    """
    dataset, eval_data_loader, train_data_loader = get_train_eval_dataloader(
        dataset, data_dir, rank, world_size, train_config, model_conf, feature_config, dataloader_config, is_recall=train_config['is_recall']
    )
    return dataset, eval_data_loader, train_data_loader

def get_optimizer(train_conf, model, learning_rate):
    """
    获取优化器和损失函数
    """
    beta = tuple(train_conf["beta"])
    optimizer_type = train_conf["optimizer_type"]
    weight_decay = train_conf["weight_decay"]
    opt_dict = {
        "adamw": torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=beta, weight_decay=weight_decay),
        "adam": torch.optim.Adam(model.parameters(), lr=learning_rate, betas=beta, weight_decay=weight_decay),
        "sgd": torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    }
    if optimizer_type.strip().lower() not in opt_dict:
        raise ValueError("Unknown optimizer_type %s" % optimizer_type)
    return opt_dict[optimizer_type.strip().lower()]

def get_optimizer_callback(train_conf) -> Callable[
    [Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]], optim.Optimizer]:
    optimizer_type = train_conf["optimizer_type"].strip().lower()
    if optimizer_type not in ("adamw", "adam", "sgd"):
        raise ValueError("Unknown optimizer_type %s" % optimizer_type)
    beta = tuple(train_conf["beta"])
    learning_rate = train_conf["learning_rate"]
    weight_decay = train_conf["weight_decay"]
    if optimizer_type == "adamw":
        return lambda params: torch.optim.AdamW(params, lr=learning_rate, betas=beta, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return lambda params: torch.optim.Adam(params, lr=learning_rate, betas=beta, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return lambda params: torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)

def time_wrap_xpu():
    if NPU_ENABLE:
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()
    return time.time()

def train_step(model, exec_cb: Callable[[int], Tuple[torch.Tensor, SequentialFeatures]],
               train_conf, export_conf, batch_id, world_size, rank, epoch, device, save_dir, profiler=None):
    """
    训练模型
    """
    model.train()
    train_costs, batch_group = [], train_conf['eval_interval']
    last_training_time = time_wrap_xpu()
    use_loss_weighted_grad = train_conf.get("use_loss_weighted_grad", False)
    is_output_loss_csv_file = train_conf.get("is_output_loss_csv_file", False)
    record_training_metrics = train_conf.get("record_training_metrics", True)
    step_list, loss_list = [], []
    step_metric_records = []
    train_times = []
    last_loss = None
    while True:
        if profiler:
            profiler.step()
        try:
            loss, _ = exec_cb(batch_id)
            last_loss = loss
            loss.backward()
        except StopIteration:
            logging.info("finished")
            break
        if is_output_loss_csv_file:
            loss_list.append(loss.detach().cpu().item())
            step_list.append(batch_id)
        if (batch_id % batch_group) == 0:
            train_cost = time_wrap_xpu() - last_training_time
            gt = 1000.0 * train_cost / batch_group
            train_times.append(gt)
            loss_value = loss.detach().float().cpu().item()
            if rank == 0:
                print("Finished train it {} of epoch {}, {:.2f} ms/it, loss {:.6f}".format(
                    batch_id, epoch, gt, loss)
                )
            if record_training_metrics:
                samples_per_second = (
                    float(train_conf.get("local_batch_size", 1)) * float(world_size) * 1000.0 / float(gt)
                    if gt > 0
                    else None
                )
                step_metric_records.append({
                    "epoch": int(epoch),
                    "step": int(batch_id),
                    "loss": float(loss_value),
                    "ms_per_step": float(gt),
                    "samples_per_second": samples_per_second,
                    "world_size": int(world_size),
                })
            if batch_id > 0:
                train_costs.append(train_cost)
            last_training_time = time_wrap_xpu()
        batch_id += 1

    if len(train_times) > 1:
        train_times.pop(0)
    if train_costs:
        logging.info(f"rank {rank} epoch {epoch}; mean cost per step: {mean(train_costs) / batch_group:.2f}s")
    else:
        logging.info(f"rank {rank} epoch {epoch}; less than {batch_group} steps")

    if is_output_loss_csv_file:
        loss_total_list = gather_all_list(loss_list, world_size=world_size)
        save_loss_csv_file(loss_total_list, step_list)

    if last_loss is None:
        last_loss = torch.zeros((), device=device)
    epoch_loss = avg_eval(torch.tensor([last_loss.detach().float().cpu().item()]).to(device), world_size=world_size)

    return batch_id, epoch_loss, train_times, step_metric_records

def save_loss_csv_file(loss_list, step_list):
    """
    保存loss.csv文件
    """
    save_dir = FLAGS.save_dir
    if not loss_list:  # 处理空列表的情况
        logging.info("loss_list is None!!!")
        pass

    step_total = len(step_list)  # 101 m/n
    all_loss_num = len(loss_list)  # 404 n
    step_length = all_loss_num / step_total  # 4 m
    # 分割成m组，每组长度为k
    sublists = [loss_list[i * step_total: (i + 1) * step_total] for i in range(int(step_length))]
    # 转置后计算每列的平均值
    score_avg_list = [sum(values) / step_length for values in zip(*sublists)]
    step_avg_dict = {k: v for k, v in zip(step_list, score_avg_list)}

    df = pd.DataFrame({
        'step': list(step_avg_dict.keys()),
        'loss': [round(v, 6) for v in step_avg_dict.values()]
    })
    # 按键排序
    df = df.sort_values('step')
    # 保存为 CSV
    save_loss_path = "%s/modelfile" % save_dir
    loss_file_name = "step_vs_loss.csv"
    loss_file_path = os.path.join(save_loss_path, loss_file_name)
    if not os.path.exists(save_loss_path):
        os.makedirs(save_loss_path, exist_ok=True)
    logging.info("Saving step_vs_loss.csv to %s", loss_file_path)
    df.to_csv(loss_file_path, index=False)


def _metrics_dir(save_dir: str) -> str:
    metrics_dir = os.path.join(save_dir, "modelfile", "run_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir


def _json_default(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _metric_to_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().cpu().item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _write_metrics_artifacts(save_dir: str, step_records, epoch_records) -> None:
    metrics_dir = _metrics_dir(save_dir)
    loss_curve_path = os.path.join(metrics_dir, "loss_curve.csv")
    epoch_metrics_path = os.path.join(metrics_dir, "epoch_metrics.csv")
    summary_path = os.path.join(metrics_dir, "metrics_summary.json")
    figure_path = os.path.join(metrics_dir, "loss_auc_curve.png")

    if step_records:
        pd.DataFrame(step_records).sort_values(["epoch", "step"]).to_csv(loss_curve_path, index=False)
    if epoch_records:
        pd.DataFrame(epoch_records).sort_values("epoch").to_csv(epoch_metrics_path, index=False)

    final_epoch = epoch_records[-1] if epoch_records else {}
    last_step = step_records[-1] if step_records else {}
    valid_qps = [
        _metric_to_float(record.get("samples_per_second"))
        for record in step_records
        if _metric_to_float(record.get("samples_per_second")) is not None
    ]
    summary = {
        "loss_curve_csv": loss_curve_path,
        "epoch_metrics_csv": epoch_metrics_path if epoch_records else None,
        "figure_png": figure_path,
        "num_loss_points": len(step_records),
        "num_eval_points": len(epoch_records),
        "final": final_epoch,
        "last_step": last_step,
        "avg_samples_per_second": float(np.mean(valid_qps)) if valid_qps else None,
        "last_samples_per_second": valid_qps[-1] if valid_qps else None,
        "auc_available": any(record.get("auc") is not None for record in epoch_records),
        "loss_decreased": (
            bool(step_records and step_records[-1]["loss"] < step_records[0]["loss"])
            if len(step_records) >= 2
            else None
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    if step_records:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax_loss = plt.subplots(figsize=(10, 5))
            loss_df = pd.DataFrame(step_records).sort_values("step")
            ax_loss.plot(loss_df["step"], loss_df["loss"], label="train loss", color="#2563eb")
            ax_loss.set_xlabel("step")
            ax_loss.set_ylabel("train loss")
            ax_loss.grid(True, alpha=0.25)

            metric_df = pd.DataFrame(epoch_records) if epoch_records else pd.DataFrame()
            if not metric_df.empty and "auc" in metric_df and metric_df["auc"].notna().any():
                ax_auc = ax_loss.twinx()
                ax_auc.plot(metric_df["step"], metric_df["auc"], label="eval AUC", color="#dc2626", marker="o")
                ax_auc.set_ylabel("eval AUC")
                lines, labels = ax_loss.get_legend_handles_labels()
                lines2, labels2 = ax_auc.get_legend_handles_labels()
                ax_loss.legend(lines + lines2, labels + labels2, loc="best")
            else:
                ax_loss.legend(loc="best")
            fig.tight_layout()
            fig.savefig(figure_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            logging.warning("Skip metrics figure because matplotlib plotting failed: %s", e)

    logging.info("Saved training metric artifacts under %s", metrics_dir)


def _safe_roc_auc(ground_truth, scores):
    if not ground_truth or not scores:
        return None
    labels = np.asarray(ground_truth)
    if np.unique(labels).size < 2:
        logging.warning("Skip AUC because eval labels contain only one class.")
        return None
    try:
        return float(roc_auc_score(labels, np.asarray(scores)))
    except Exception as e:
        logging.warning("Skip AUC because roc_auc_score failed: %s", e)
        return None


def save_model_state_dict(model, save_dir, export_conf):
    model_file_name = "model_hstu.pth"
    export_dir = os.path.join(save_dir, export_conf["save_dir_name"])
    model_path = os.path.join(export_dir, model_file_name)

    # 路径不存在则创建
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
        logging.info("Created directory: %s", export_dir)

    model.eval()
    logging.info("Saving model to %s", model_path)
    torch.save(model.state_dict(), model_path)

def save_model(model, 
               save_dir,
               model_input_export_list, 
               export_conf, 
               feature_conf,
               feature_map_dir_or_path, 
               cut_off_timestamp, 
               device, 
               feature_map=None, is_recall=False):
    """
    保存模型
    """
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    if not os.path.exists(os.path.join(save_dir, export_conf["save_dir_name"])):
        os.makedirs(os.path.join(save_dir, export_conf["save_dir_name"]), exist_ok=True)
        
    save_model_state_dict(model, save_dir, export_conf)
    
    if is_recall:
        return
    # ["fake", "real"]两种模式，即导出的候选集是真实数据还是随机生成的数据
    export_mode = export_conf.get("export_mode", "fake")

    model_input_export_list_new = []

    export_batch_size = export_conf.get('export_batch_size', 1)
    num_rerank = export_conf.get("num_rerank", 256)
    shape_checked = False

    logging.info("export_batch_size is %s", export_batch_size)
    logging.info("num_rerank is %s", num_rerank)
    for inputs in model_input_export_list:
        for k in inputs:
            inputs[k] = inputs[k][:export_batch_size]
            if export_mode == "real":
                if "candidate_" in k:
                    _, real_length = inputs[k].size()
                    inputs[k] = torch.nn.functional.pad(inputs[k], (0, num_rerank - real_length))
            elif export_mode == "fake":
                if k not in feature_conf['item_feature_columns']:
                    continue
                if "candidate_" in k:
                    if k == "candidate_timestamps":
                        len_candidate_timestamps_fake = num_rerank * export_batch_size
                        candidate_timestamps_fake = [int(cut_off_timestamp + 86399)] * len_candidate_timestamps_fake
                        candidate_timestamps_fake = torch.tensor(candidate_timestamps_fake, dtype=torch.int64).view(
                            export_batch_size, -1).to(device)
                        inputs[k] = candidate_timestamps_fake
                    elif k == "candidate_labels":
                        candidate_labels_fake = torch.randint(2, (export_batch_size, num_rerank), dtype=torch.int64).to(device)
                        inputs[k] = candidate_labels_fake
                    elif k == "candidate_item_id":
                        candidate_feature_count = feature_conf["item_feature_columns"][feature_conf["infer_items_key"]][
                                                        "feature_count"]
                        candidate_item_id_fake = torch.randint(candidate_feature_count, (export_batch_size, num_rerank), dtype=torch.int64).to(device)
                        inputs[k] = candidate_item_id_fake
                    else:
                        candidate_dtype = feature_conf["item_feature_columns"][k.replace("candidate_", "")]["dtype"]
                        if candidate_dtype == "int":
                            candidate_feature_count = feature_conf["item_feature_columns"][k.replace("candidate_", "")][
                                                        "feature_count"]
                            candidate_feat_fake = torch.randint(candidate_feature_count,
                                                                (export_batch_size, num_rerank), dtype=torch.int64).to(device)
                        elif candidate_dtype == "con":
                            candidate_feat_fake = torch.rand((export_batch_size, num_rerank), dtype=torch.float32).to(device)
                        inputs[k] = candidate_feat_fake
            else:
                logging.error("export_mode in export_conf should be either real or fake")
       
        input_keys = list(inputs.keys())
        for k in input_keys:
            if k.replace("candidate_", "") in feature_conf["item_feature_columns"]:
                enabled = feature_conf["item_feature_columns"][k.replace("candidate_", "")].get("enabled", True)
                if not enabled:
                    inputs.pop(k, None)
            elif k in feature_conf["user_feature_columns"]:
                enabled = feature_conf["user_feature_columns"][k].get("enabled", True)
                if not enabled:
                    inputs.pop(k, None)
            # 以下这些key值不在音乐业务里使用
            elif k in ["candidate_item_id"] and dataset_name == "music-scalingraw-rank":
                inputs.pop(k, None)
        if not shape_checked:
            for k,v in inputs.items():
                print(k, "|", to_numpy(v).shape)
            shape_checked = True
        model_input_export_list_new.append(inputs)
    infer_item_id_name = "candidate_" + feature_conf["infer_items_key"]
    y_true_eval_all = []
    y_score_eval_all = []
    output_all = None
    input_all = {}
    input_keys = list(inputs.keys())
    candidate_label_csv = []
    score_csv= []
    for inputs in model_input_export_list_new:
        model_out = model(inputs)["rerank_score"]
        candidate_labels = inputs["candidate_labels"]
        candidate_ids = inputs[infer_item_id_name]
        y_true_eval = to_numpy(candidate_labels[candidate_ids != 0])
        y_score_eval = to_numpy(model_out[candidate_ids != 0])
        for candidate_label_row, candidate_id_row, model_out_row in zip(candidate_labels, candidate_ids, model_out):
            candidate_label_csv.append("^".join(str(x) for x in to_numpy(candidate_label_row[candidate_id_row != 0])))
            score_csv.append("^".join(str(x) for x in to_numpy(model_out_row[candidate_id_row != 0])))
        model_output = to_numpy(model_out)
        inputs = {k: to_numpy(inputs[k]) for k in inputs}
        y_true_eval_all.append(y_true_eval)
        y_score_eval_all.append(y_score_eval)
        if output_all is None:
            output_all = model_output
        else:
            output_all = np.concatenate([output_all, model_output], axis=0)
        for k in input_keys:
            if not k in input_all:
                input_all[k] = inputs[k]
            else:
                input_all[k] = np.concatenate([input_all[k], inputs[k]], axis=0)
    is_compute_export_auc = export_conf.get('compute_auc', False)
    if is_compute_export_auc:
        y_true_eval_all = np.concatenate(y_true_eval_all, axis=None)
        y_score_eval_all = np.concatenate(y_score_eval_all, axis=None)
        export_pctr = np.mean(y_true_eval_all)
        export_pctr_gt_ctr = np.mean(y_score_eval_all)
        export_auc = roc_auc_score(y_true_eval_all, y_score_eval_all)
        logging.info("export auc is %s", export_auc)
    with open("%s/%s/pth_input" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
        pickle.dump(input_all, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open("%s/%s/pth_output" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
        pickle.dump(output_all, fp, protocol=pickle.HIGHEST_PROTOCOL)
    uid_csv = input_all["uid"].tolist()
    need_export_csv = export_conf.get("need_export_csv", False)
    if need_export_csv:
        export_csv = pd.DataFrame({"column_id": uid_csv, "label_id": candidate_label_csv, "score": score_csv})
        with open("%s/%s/result.csv" % (save_dir, export_conf["save_dir_name"]), 'wb') as fp:
            export_csv.to_csv(fp, index=False)

def save_feature_map(export_conf, save_dir, feature_map_dir_or_path, feature_map=None):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists("%s/%s.config" % (save_dir, export_conf["save_dir_name"])):
        os.mkdir("%s/%s.config" % (save_dir, export_conf["save_dir_name"]))
    if feature_map is not None:
        feature_dict = feature_map
    else:
        feature_map_files = os.listdir(feature_map_dir_or_path)
        feature_dict = dict()
        for feature_map_file in feature_map_files:
            feature_map_df = pd.read_orc(os.path.join(feature_map_dir_or_path, feature_map_file))
            indices_to_remove = [i for i, value in enumerate(feature_map_df['feature_name']) if 'user_id' in value]
            feature_map_df = feature_map_df.drop(indices_to_remove)
            feature_name = feature_map_df['feature_name'].tolist()
            feature_id = feature_map_df['feature_id'].tolist()
            feature_id = [int(x) for x in feature_id]
            feature_dict.update(zip(feature_name, feature_id))
    with open("%s/%s.config/feature_map.json" % (save_dir, export_conf["save_dir_name"]), 'w', encoding="utf-8") as fp:
        json.dump(feature_dict, fp, indent=4, ensure_ascii=False)
    logging.info("Saved feature map to %s/%s.config/feature_map.json", save_dir, export_conf["save_dir_name"])

def save_profiler_to_execl(prof, folder_name):
    if NPU_ENABLE:
        return
    key_averages_events = prof.key_averages()
    if key_averages_events:
        print("可用的 Event 属性列表:")
        print(dir(key_averages_events[0]))
    profiler_data = []
    for event in key_averages_events:
        profiler_data.append({
            'Name': event.key,
            'CPU Time Total (us)': event.cpu_time_total,
            'CUDA Time Total (us)': event.device_time_total,  # <-- 修正
            'CPU Time Self (us)': event.self_cpu_time_total,
            'CUDA Time Self (us)': event.self_device_time_total,  # <-- 修正
            'CPU Memory Self (B)': event.self_cpu_memory_usage,
            'CUDA Memory Self (B)': event.self_device_memory_usage,  # <-- 修正
            'Number of Calls': event.count,
        })
    df = pd.DataFrame(profiler_data)
    # 5. (可选) 对数据进行排序，方便分析
    # 按 CUDA 总时间降序排列
    df_sorted = df.sort_values(by='CUDA Time Total (us)', ascending=False)

    # 6. 导出到 Excel 文件
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_excel_path = os.path.join(folder_name, f'{time_str}_profiler_results.xlsx')
    df_sorted.to_excel(output_excel_path, index=False)

def save_e2e_to_execl(e2e_data, folder_name):
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_csv_path = os.path.join(folder_name, f"{time_str}_e2e.csv")
    df = pd.DataFrame(e2e_data)
    df.to_csv(output_csv_path, index=False)

def patch_graph(model):
    model.graph_prepared = False
    model.static_model_inputs = {}
    model.static_outputs = None
    if not NPU_ENABLE:
        # CUDA process contains pure CPU operations (grouped gemm) adn cannot be fully integrated into the graph.
        return
    else:
        model.stream = torch.npu.Stream()
        model.graph = torch.npu.NPUGraph()
    ori_encode = model.encode_static
    
    def encode_graph(model_inputs):
        if not model.graph_prepared:
            model.static_model_inputs = {k: v.clone() for k, v in model_inputs.items()} if model_inputs is not None else None
            # warmup
            for _i in range(3):
                with torch.no_grad():
                    _ = ori_encode(model.static_model_inputs)
            with torch.npu.graph(model.graph, None, model.stream):
                model.static_outputs = ori_encode(model.static_model_inputs)
            model.graph_prepared = True
        else:
            if model_inputs is not None:
                for k in model_inputs.keys():
                    model.static_model_inputs[k].copy_(model_inputs[k])
        model.graph.replay()
        return model.static_outputs
    model.encode_static = encode_graph

def get_hook(name, captured_data, module):
    def hook(model, input, output):
        layer_name = f"{name} ({module.__class__.__name__})"
        captured_data[layer_name] = {
            "type": "named_module",
            "input": to_clean_cpu_tensor(input),
            "output": to_clean_cpu_tensor(output),
            "parameter": to_clean_cpu_tensor(module.state_dict()),
        }
        #logging.info(f"Hook captured data from: {layer_name}")
    return hook

def register_forward_output_hook(model, captured_data):
        for name, layer in model.named_modules():
            if not list(layer.children()):
                layer.register_forward_hook(get_hook(name, captured_data, layer))

def save_torch_data(input, output, layer_data=None, filename="infer_result",
                        layer_result_dir="../infer_result/"):
                        
        if not os.path.exists(layer_result_dir):
            os.makedirs(layer_result_dir, exist_ok=True)
            logging.info("Created directory: %s", layer_result_dir)

        total_infer_result = {
            "tokenmixer_large": {
                "type": "model",
                "input": to_clean_cpu_tensor(input),
                "output": to_clean_cpu_tensor(output),
            }
        }
        logging.info(f"\n--- saving data... ---")
        os.makedirs(layer_result_dir, exist_ok=True)
        result_path = os.path.join(layer_result_dir, filename)
        result_path = get_unique_path(result_path)
        torch.save(total_infer_result, result_path)
        logging.info(f"\n--- infer output data saved to: {result_path} ---")
        if layer_data is not None:
            filename_layer = filename + "layer"
            result_path_layer = os.path.join(layer_result_dir, filename_layer)
            torch.save(layer_data, result_path_layer)
            logging.info(f"\n--- layer infer output data saved to: {result_path_layer} ---")

def load_feature_map_cached(txt_path: str, target_field: str, cache_path: str = None) -> dict:
    if cache_path is None:
        cache_path = f"{txt_path}.{target_field}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    feature_map = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            key, feature_id_str = line.split('\t')
            field_name, original_value = key.split(',', 1)
            if field_name == "App.Id":
                feature_map[int(feature_id_str)] = original_value
    with open(cache_path, 'wb') as f:
        pickle.dump(feature_map, f)
    return feature_map

def save_item_embs(model, exec_cb: Callable[[int], Tuple[torch.Tensor, SequentialFeatures]],
            feature_conf, feature_map, feature_name, feature_map_path, train_conf, save_dir, feature_map_dir_or_path,
            export_conf, world_size, rank, save_after_eval, device, dataset, all_item_ids):
    """
    保存某个特征如 App.Id的item embedding
    每一行为 original_value, embedding
    """
    model.eval()
    logging.info(f"Generating item embeddings for feature: {feature_name}")

    feature_id_to_original = load_feature_map_cached(
        txt_path=feature_map_path,
        target_field=feature_name
    )

    emb_layer = model.module.embedding_module._item_info_embs[feature_name]
    all_embeddings: torch.Tensor = emb_layer.weight.detach().to(device).cpu()
    total_items = all_embeddings.size(0)

    group_keys_all = {
        'original_value': [],
        'embedding': []
    }

    for fid in range(total_items):
        if fid == 0 or fid not in feature_id_to_original:
            continue
        original_value = feature_id_to_original[fid]
        emb = all_embeddings[fid].tolist()
        emb = [','.join([str(d) for d in t]) for t in emb]
        group_keys_all['original_value'].append(original_value)
        group_keys_all['embedding'].append(emb)
    save_path = os.path.join(save_dir, "modelfile")
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f"item_embedding_{feature_name}.csv")
    logging.info(f"Saving item embeddings to {output_file}")
    if rank == 0:
        with open(output_file, "w") as f:
            for a,b in zip(group_keys_all['original_value'], group_keys_all['embedding']):
                f.write(f"{a};{b}")
    logging.info(f"Done saving item embedding to {output_file}")

def save_user_embs(model, exec_cb: Callable[[int], Tuple[torch.Tensor, SequentialFeatures]],
            feature_conf, feature_map, train_conf, save_dir, feature_map_dir_or_path,
            export_conf, world_size, rank, save_after_eval, device, dataset, all_item_ids):
    """
    评估模型
    """
    model.eval()
    logging.info(f"Generating user embeddings for rank ... {rank}")
    group_keys_all = {
        'uid': [],
        'embeddings' : []
    }
    save_iter, part_iter = 0, 0
    last_eval_time = time.perf_counter()

    def write_embs(rank, partition, group_keys, save_dir):
        save_path = f"{save_dir}/user_embeddings/"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        user_embedding_file_name = f"user_embedding_part_{rank}_{partition}.csv"
        embedding_path_name = os.path.join(save_path, user_embedding_file_name)
        logging.info("Saving user embeddings to %s", embedding_path_name)
        user_embedding_df = pd.DataFrame({
            **group_keys
        })
        user_embedding_df
        user_embedding_df.to_csv(embedding_path_name,header=False,index=False,sep='\t')
    
    while True:
        try:
            scores, seq_features = exec_cb(save_iter)
        except StopIteration:
            logging.info("finished")
            break
        
        model_input = seq_features.past_payloads
        model_input['past_lengths'] = seq_features.past_lengths
        model_input['past_ids'] = seq_features.past_ids
        B, N = seq_features.past_ids.size()
        
        user_embeddings : torch.Tensor = model.module.encode(
            past_lengths=seq_features.past_lengths,
            past_ids=seq_features.past_ids,
            past_embeddings=None,
            past_payloads=seq_features.past_payloads,
            model_inputs=model_input,
        )
        def convert_uid_tensor_to_hex(uid_list_hex): 
            l = [] 
            for uid_hex in uid_list_hex:
                s = ''
                for i, h in enumerate(uid_hex):
                    str_seg = hex(h)[2:]
                    if i < 4:
                            str_seg = ('0' * (15 - len(str_seg))) + str_seg
                            s += str_seg
                    elif i == 4:  
                        str_seg = ('0' * (4 - len(str_seg))) + str_seg
                        s += str_seg
                l.append(s.upper())
            return l

        user_embeddings_cpu = user_embeddings.detach().cpu().tolist()
        uid_list_hex = convert_uid_tensor_to_hex(seq_features.uid.tolist())
        group_keys_all['uid'] +=  uid_list_hex
        user_embeddings_cpu = ['^'.join([str(d) for d in t]) for t in user_embeddings_cpu]
        group_keys_all['embeddings'] += user_embeddings_cpu

        if (save_iter % train_conf["eval_interval"]) == 0:
                cost = time.perf_counter() - last_eval_time
                logging.info(
                    f"rank {rank}; batch-stat (SAVE USER EMBS): step {save_iter} (SAVE USER EMBS in {cost:.2f}s)")
                last_eval_time = time.perf_counter()
                if part_iter % 10 == 0 and part_iter != 0:
                    logging.info(f'dumping user file rank={rank} num_partition={part_iter//10}')
                    write_embs(rank, part_iter//10, group_keys_all, save_dir)
                    group_keys_all = {
                        'uid': [],
                        'embeddings': []
                    }
                part_iter += 1
                torch.distributed.barrier()  # sync time taken
                
        del user_embeddings
        del model_input
        save_iter += 1
    write_embs(rank, part_iter//10, group_keys_all, save_dir)


def latency_summary_rank0(latency_list, batch_size, model_name, device_name, inductor_flag):
    if latency_list is None or len(latency_list) == 0:
        print("No latency data collected, please check sample_statistics_num value effectively collects latency data.")
        return None

    ave_step = np.array(latency_list)
    avg_time = sum(ave_step) / len(ave_step)
    ave_step.sort()

    if len(ave_step) > 0:
        p90 = np.percentile(ave_step, 90, method="lower")
        p95 = np.percentile(ave_step, 95, method="lower")
        p99 = np.percentile(ave_step, 99, method="lower")
        p999 = np.percentile(ave_step, 99.9, method="lower")
    QPS = int(batch_size*1000.0/avg_time)     # batch_size * world_size * 10 / avg_time
    e2e_result = {
        "Batch_size": batch_size,
        "model_name": model_name,
        "QPS": QPS,
        "AVG Latency": avg_time,
        "P999 Latency": p999,
        "P99 Latency": p99,
        "P95 Latency": p95,
        "P90 Latency": p90,
    }

    model_detail_info = device_name + "_" + model_name + "_" + inductor_flag
    output_str = "performance: " + model_detail_info + "_" + str(e2e_result)
    print(f"The final performance result: {output_str}")

    os.makedirs(f"../save_results_{device_name}/", exist_ok=True)
    with open(f"../save_results_{device_name}/performance_result.txt", "a", encoding="utf-8") as f:
        f.write(output_str + "\n")
    return avg_time

def gather_latency_data(local_data, world_size, device):
    """
    收集各 rank 的 latency 数据，返回 list[list[float]]
    """
    if world_size <= 1:
        return [local_data]
    local_tensor = torch.tensor(local_data, dtype=torch.float32, device=device)
    # all_gather 自动同步
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    return [g.cpu().tolist() for g in gathered]

def flattn_last_n(maxtrix, n):
    if n <= 0:
        return []
    return [item for sub in maxtrix for item in sub[-n:]]

def eval_fn(config, data_dir, save_dir, feature_map_dir_or_path, period, already_init=False) -> None:
    """
    单独评估函数
    """
    # 1. 初始化随机数、加速器，配置文件
    # 1.1 这一部分可以通用
    common_config = config["common_hp"]
    init_random_seed(common_config)
    device, rank, local_rank, world_size, node_num = init_ddp_info_params(already_init)

    train_conf = common_config['train_conf']
    feature_conf = common_config['feature_conf']
    if FLAGS.eval_batch_size is not None:
        train_conf['eval_batch_size'] = FLAGS.eval_batch_size
    learning_rate = train_conf['learning_rate']
    lr_scaling = train_conf['lr_scaling']
    export_conf = common_config['export_conf']
    data_loader_conf = common_config['data_loader_conf']
    model_conf = common_config['model_conf']
    dataset_name = data_loader_conf["dataset_name"]
    feature_conf["period"] = period
    train_conf['learning_rate'] = init_learning_rate(learning_rate, lr_scaling, world_size)
    filter_invalid_ids = train_conf.get('filter_invalid_ids', False)
    logging.info(f'debug filter invalid ids {filter_invalid_ids}')
    
    logging.info("Testing model on rank %s (local rank: %s); device: %s; world_size: %s;",
                 rank, local_rank, device, world_size)

    init_torch_config(local_rank, train_conf)

    # 1.2 这一部分根据业务数据格式，需要用特定的方式编辑feat_conf和model_conf

    feature_conf, model_conf, feature_map = refine_feat_and_model_conf(dataset=dataset_name,
                                                                       feature_conf=feature_conf,
                                                                       model_conf=model_conf,
                                                                       feature_map_dir_or_path=feature_map_dir_or_path,
                                                                       model_cfg=config[Const.MODEL_CFG])
    # 2. 初始化数据集
    dataset, eval_data_loader, _ = get_data_loaders(dataset=dataset_name,
                                                                    data_dir=data_dir,
                                                                    rank=rank,
                                                                    world_size=world_size,
                                                                    train_config=train_conf,
                                                                    model_conf=model_conf,
                                                                    feature_config=feature_conf,
                                                                    dataloader_config=data_loader_conf)

    ModelRegistry.register_all_modules(modul_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "modeling/generic/sequential_v2")))
    model = ModelInitializer.init(gr_module_cfg=config)
    if hasattr(model, 'negative_sampler') and hasattr(model.negative_sampler, 'set_all_item_ids') and train_conf.get('filter_invalid_ids', False):
        model.negative_sampler.set_all_item_ids(dataset.all_item_ids)

    # 6.加载模型(精度对齐)
    load_pretrain_model = False if int(os.environ.get("LOAD_PRETRAIN_MODEL", "0")) == 0 else True
    if load_pretrain_model:
        model_file_name = "model_hstu.pth"
        model_path = os.path.join(save_dir, export_conf["save_dir_name"], model_file_name)
        model_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state_dict, False)
        logging.info(f"Loaded pretrain model from {model_path}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Total number of parameters: {total_params:.2f}M")
    total_params = sum(p.numel() for x, p in model.named_parameters() if '_emb' not in x) / 1e6
    logging.info(f"The number of parameters (exl. emb): {total_params:.2f}M")

    if model_conf.get("root_model_type") == "GRModelEp":
        from modeling.generic.sequential_v2.torchrec_trainer import (assemble_model_executable_callback_ep,
                                                                  feed_datas_to_exec_callback_ep)
        model, exec_cb = assemble_model_executable_callback_ep(
            model,
            None,
            world_size,
            node_num,
            device,
            train_conf,
            feature_conf,
        )
        feed_func = feed_datas_to_exec_callback_ep
    else:
        model, exec_cb = assemble_model_executable_callback(
            model,
            None,
            local_rank,
            device,
            train_conf,
            feature_conf,
        )
        feed_func = feed_datas_to_exec_callback

    _, eval_exec_cb = feed_func(exec_cb, train_conf['is_recall'], None, eval_data_loader)

    if dataset_name == "ag-rank":
        # 保存统计结果
        data_target.save_data(rank)
    # 6. 测试模型
    try:
        evaluate_step(
            model=model,
            exec_cb=eval_exec_cb,
            feature_conf=feature_conf,
            feature_map=feature_map,
            train_conf=train_conf,
            save_dir=save_dir,
            feature_map_dir_or_path=feature_map_dir_or_path,
            export_conf=export_conf,
            world_size=world_size,
            rank=rank,
            save_after_eval=True,
            device=device,
            dataset=dataset_name,
            all_item_ids=dataset.all_item_ids,
            filter_invalid_ids=filter_invalid_ids,
        )
    except Exception as e:
        logger.exception("evaluate_step failed: %s", e)
        raise
    try:
        save_model_state_dict(model, save_dir, export_conf)
    except Exception as e:
        logger.exception("save_model_state_dict failed: %s", e)
        raise

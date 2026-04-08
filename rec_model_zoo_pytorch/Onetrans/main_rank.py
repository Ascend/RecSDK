from gr_model import Const, OneTrans
import random
import numpy as np
import torch
import os
import logging
import torch.distributed as dist
import subprocess
from math import sqrt
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional
from torch.utils.data import IterableDataset
import pandas as pd
import ast
from torchrec.streamable import Pipelineable
import argparse
import yaml
from datetime import datetime
import sys
import time
import json
from tqdm import tqdm
from dataclasses import dataclass
from check_result import IoChecker

NPU_ENABLE = True
if os.environ.get("NPU_FLAG", "True") == "False":
    NPU_ENABLE = False
else:
    import torch_npu
    from torch_npu.profiler import profile


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument("--test_data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--period", type=str)
    parser.add_argument("--tensorboard_log_dir", type=str)
    parser.add_argument("--is_train", default=True)
    parser.add_argument("--save_user_emb", default=False)
    parser.add_argument("--open_megatron", default=False)
    parser.add_argument("--data_path", default='', type=str)
    args = parser.parse_args()
    return args


def extra_init_args(parser):
    group = parser.add_argument_group('Distributed GR Arguments')
    group.add_argument("--config_file", type=str)
    group.add_argument("--data_dir", type=str)
    group.add_argument("--save_dir", type=str)
    group.add_argument("--period", type=str)
    group.add_argument("--tensorboard_log_dir", type=str)
    group.add_argument("--is_train", default=True)
    group.add_argument("--save_user_emb", default=False)
    group.add_argument("--open_megatron", default=True)
    group.add_argument("--use_tokenizer", default=False)
    group.add_argument("--data_path", default='')
    return parser


def read_yml(file_path, use_e=False):
    with open(file_path, 'r', encoding='UTF-8') as yml_file:
        result = yaml.safe_load(yml_file)
    return result


def get_config(file_path):
    config_dict = read_yml(file_path, use_e=True)

    return config_dict


def init_torch_config(local_rank, train_conf):
    """
    Init distributed configs
    """
    use_tf32 = train_conf.get("use_tf32", False)
    if NPU_ENABLE:
        torch_npu.npu.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    logging.info("cuda.matmul.allow_tf32: %s", use_tf32)
    logging.info("cudnn.allow_tf32: %s", use_tf32)


def get_conf(common_config):
    return common_config['train_conf'], common_config['feature_conf'], \
        common_config['data_loader_conf'], common_config['model_conf']


def init_random_seed(config):
    random_seed = config.get('seed_conf', {}).get("global_seed", '1234')
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if NPU_ENABLE:
        torch_npu.npu.manual_seed(random_seed)
        torch_npu.npu.manual_seed_all(random_seed)


def log_pid_localrank_cpu(rank):
    pid = os.getpid()

    try:
        result = subprocess.run(['/usr/bin/taskset', '-cp', str(pid)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        cpu_binding = result.stdout.strip()
    except Exception as e:
        cpu_binding = f"Unable to obtain CPU binding info: {e}"

    logging.info(f"[PID {pid}] Local Rank: {rank} | CPU binding: {cpu_binding}")

    if rank == 0:
        try:
            lscpu_result = subprocess.run(['/usr/bin/lscpu'],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          text=True)
            logging.info("[lscpu output]:")
            logging.info(lscpu_result.stdout.strip())
        except Exception as e:
            logging.info(f"Failed to obtain lscpu info: {e}")

        try:
            npu_result = subprocess.run(['/path/to/npu-smi', 'info', '-t', 'topo'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
            logging.info("[npu-smi info -t topo output]:")
            logging.info(npu_result.stdout.strip())
        except Exception as e:
            logging.info(f"Failed to obtain npu-smi topo info: {e}")


def init_ddp_info_params(already_init=False):
    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_num = int(os.environ.get("NNODES", 1))
    logging.info(f"tcp://{addr}:{port}, nnodes={node_num}, rank={rank_id}")

    # initialize the process group
    if not already_init and NPU_ENABLE:
        # initialize the process group
        dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id,
                                world_size=world_size)
    # log_pid_localrank_cpu(rank_id)

    if NPU_ENABLE:
        return f"npu:{local_rank}", rank_id, local_rank, world_size, node_num
    else:
        return f"cuda:{local_rank}", rank_id, local_rank, world_size, node_num


def populate_args(args, **kwargs):
    for key, value in kwargs.items():
        args.__setattr__(key, value)


def init_learning_rate(learning_rate, lr_scaling, world_size):
    lr_scaling = lr_scaling.strip().lower()
    if lr_scaling == "linear":
        learning_rate *= world_size
    elif lr_scaling == "sqrt":
        learning_rate *= sqrt(world_size)
    else:
        raise ValueError("'%s' is not a supported scaling strategy for the learning rate." % lr_scaling)
    return learning_rate


def set_tensorboard_writer(rank, tensorboard_log_dir):
    if rank == 0 and tensorboard_log_dir:
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logging.info(f"Rank {rank}: writing logs to {tensorboard_log_dir}")
    else:
        writer = None
        logging.info(f"Rank {rank}: disabling summary writer")
    return writer


def refine_feat_and_model_conf(dataset: str,
                               feature_conf: Dict,
                               model_conf: Dict,
                               model_cfg, **kwargs):
    if dataset in ["amazon-recall", "movielens-1m", "kuairand-recall", "kr-rank-dlrm", "kr-rank-gr"]:
        feature_conf["cut_off_time"] = 0
    else:
        raise NotImplementedError("Unknown dataset")

    model_conf["root_model_type"] = model_cfg[Const.MODULE_NAME]
    return feature_conf, model_conf


class MultiCSVIterator:
    """Iterator to read multiple CSV files"""

    def __init__(self, file_paths: List[str], sep: str, rank: int, world_size: int, column_names, inner_delim,
                 itemid_column_name='item_id', file_format='csv', chunksize=10000000,
                 use_column_names=False, use_cols=None, shuffle=False, epoch_num=0, is_dynamic_batch=False,
                 is_load_balance=False) -> None:
        """
        Initialize MultiCSVIterator

        :param file_paths: filepaths for dataset
        :param sep: seperators for csv documents
        :param rank: rank of NPU
        :param world_size: number of NPUs
        :param column_names: names of columns in CSV files
        :param inner_delim: delimiter for columns
        :param itemid_column_name: item_id column name
        :param use_column_names: if not None, overwrite column names in csv files using this list
        :param chunksize: number of data per chunk
        :param use_column_names: whether use given column names for header or column names from csv files
        :param shuffle: whether to perform dataset shuffle
        :param epoch_num: current epoch (for shuffle)
        :param is_dynamic_batch: whether to remove last batch from data (happens typically when
        number of entries not divisible by batchsize)
        """

        self.file_paths = file_paths
        self.sep = sep
        self.rank = rank
        self.world_size = world_size
        self.node_num = int(os.environ.get("NNODES", 1))
        self.current_file_index = 0
        self.chunk_iterator = iter([])
        self.chunk_data = None
        self.total_read_count = 0
        self.is_load_balance = is_load_balance

        self.column_names = column_names
        self.inner_delim = inner_delim

        self.itemid_column_name = itemid_column_name
        self.worker_info = torch.utils.data.get_worker_info()
        self.file_format = file_format
        self.shuffle = shuffle
        if self.is_load_balance:
            self.increment = 1
            self.current_data_index = 0
        else:
            self.increment = self.world_size
            self.current_data_index = self.rank
        self.chunk_size = chunksize
        self.use_column_names = use_column_names

        self.current_file = None
        self.current_chunk = None
        self.valid_data_size = 0
        self.use_cols = use_cols
        self.epoch_num = epoch_num
        self.is_dynamic_batch = is_dynamic_batch
        self.current_chunk_idx = 0
        self.shuffled_indices = [i for i in range(self.valid_data_size)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_file is None:
            worker_info = torch.utils.data.get_worker_info()
            if self.current_file_index >= len(self.file_paths):
                logging.info('worker_id %s finished data iteration ', worker_info.id)
                raise StopIteration
            if self.file_format == 'csv':
                self.current_file = pd.read_csv(self.file_paths[self.current_file_index],
                                                sep=self.sep, chunksize=self.chunk_size,
                                                names=self.column_names if self.use_column_names else None,
                                                usecols=self.use_cols)
            elif self.file_format == 'gz':
                self.current_file = pd.read_csv(self.file_paths[self.current_file_index],
                                                sep=self.sep, chunksize=self.chunk_size,
                                                names=self.column_names if self.use_column_names else None,
                                                usecols=self.use_cols)
            self.current_chunk_idx = 0

        if self.current_chunk is None:
            try:
                if self.file_format == 'csv':
                    self.current_chunk = self.current_file.__next__()

                if self.is_dynamic_batch:
                    self.valid_data_size = len(self.current_chunk)
                else:
                    self.valid_data_size = len(self.current_chunk) - len(self.current_chunk) % self.world_size
                if self.shuffle:
                    self.shuffled_indices = [i for i in range(self.valid_data_size)]
                    random.seed(self.epoch_num)
                    random.shuffle(self.shuffled_indices)
            except StopIteration:
                self.current_data_index = self.rank
                if self.is_load_balance:
                    self.current_data_index = 0
                self.current_file = None
                self.current_file_index += 1
                return self.__next__()

        if self.current_data_index < self.valid_data_size:
            data_idx = self.current_data_index
            if self.shuffle:
                data_idx = self.shuffled_indices[self.current_data_index]
            data = self.current_chunk.iloc[data_idx]
            self.current_data_index += self.increment
            return data
        else:
            self.current_data_index = self.rank
            if self.is_load_balance:
                self.current_data_index = 0
            self.current_chunk = None
            return self.__next__()
class RankDLRMDataloader(IterableDataset):
    def __init__(
            self,
            padding_length,
            user_feat_config: dict,
            item_feat_config: dict,
            seq_feat_config: dict,
            dataset_train_dir: str = None,
            dataset_valid_dir: str = None,
            rank: int = 0,
            world_size: int = 1,
            is_train: bool = True,
            file_format='csv',
            sep: str = ',',
            exclude_features=[],
            num_rerank=256,
            infer_label_key: str = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.sep = sep
        self.is_train = is_train
        self.exclude_features = exclude_features
        self.infer_label_key = infer_label_key

        if is_train and dataset_train_dir is None:
            raise ValueError('dataset_train_dir should be configured when it is training')
        if not is_train and dataset_valid_dir is None:
            raise ValueError('dataset_valid_dir should be configured when it is not training')

        if is_train:
            train_files = os.listdir(dataset_train_dir)
            random.shuffle(train_files)
            self.files = [dataset_train_dir + '/' + f for f in train_files if f.endswith(".csv") or f.endswith(".gz")]
        else:
            self.num_rerank = num_rerank
            valid_files = os.listdir(dataset_valid_dir)
            random.shuffle(valid_files)
            self.files = [dataset_valid_dir + '/' + f for f in valid_files if
                          f.endswith(".csv") or f.endswith(".gz")]

        self.multi_csv_iterator = None
        self.file_format = file_format

        self.seq_feat_config = seq_feat_config
        self.item_feat_config = item_feat_config
        self.user_feat_config = user_feat_config
        self.seq_feat_lengths = [x["length"] for _, x in seq_feat_config.items()][0]
        self.shuffle = False
        self.epoch_num = 0
        self.is_dynamic_batch = False
        self.is_load_balance = False
        self.max_seq_len = padding_length

    def _truncate_or_pad_sequence(self, tensor, target_len):

        current_len = tensor.size(1)
        if current_len > target_len:
            # 截断：取最左边 target_len 个（假设最新数据在前）
            tensor = tensor[:, :target_len]
        elif current_len < target_len:
            pad_size = target_len - current_len
            tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)

        return tensor

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            worker_info = torch.utils.data.get_worker_info()
            self.multi_csv_iterator = MultiCSVIterator(self.files, self.sep, self.rank, self.world_size,
                                                       None, ',', itemid_column_name='',
                                                       shuffle=self.shuffle, epoch_num=self.epoch_num,
                                                       is_dynamic_batch=self.is_dynamic_batch,
                                                       is_load_balance=self.is_load_balance)

    def __iter__(self):
        it = map(self.load_item, self.multi_csv_iterator)
        return it

    def pad_or_truncate_item_feature(self, feat_value: torch.tensor, max_length: int, dtype):
        if dtype == "int" or dtype == "con":
            num_valid_test_items = feat_value.size(-1)
        else:
            num_valid_test_items = feat_value.size(0)
        if num_valid_test_items < max_length:
            if dtype == "int" or dtype == "con":
                feat_value = torch.nn.functional.pad(feat_value, (0, max_length - num_valid_test_items))
            else:
                feat_value = torch.nn.functional.pad(feat_value, (0, 0, 0, max_length - num_valid_test_items))
        return feat_value, num_valid_test_items

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        ret = {}
        for key, value in data.items():

            if key in self.exclude_features:
                continue

            else:
                if isinstance(value, str):
                    value = ast.literal_eval(value)
                elif isinstance(value, np.generic):
                    value = int(value)

                if isinstance(value, list):
                    if isinstance(value[0], int):
                        value = torch.tensor(value, dtype=torch.int64).reshape(1, -1)
                    elif isinstance(value[0], float):
                        value = torch.tensor(value, dtype=torch.float32).reshape(1, -1)

                else:
                    if isinstance(value, int):
                        value = torch.tensor(value, dtype=torch.int64).reshape(1, -1)
                    elif isinstance(value, float):
                        value = torch.tensor(value, dtype=torch.float32).reshape(1, -1)

                    else:
                        logging.error("please check the data type value of %s, which is %s", key, value)

            if key in ["history_timestamps", "history_item_ids"]:
                value = torch.flip(value, dims=[1])

            ret[key] = value

        if "history_item_ids" in ret:
            ret["history_item_ids"] = self._truncate_or_pad_sequence(
                ret["history_item_ids"], self.max_seq_len
            )

        ret["history_lengths"] = torch.count_nonzero(ret["history_item_ids"]).reshape(1, -1)
        ret["uid"] = torch.tensor(0, dtype=torch.int64).reshape(1, -1)
        ret["labels"] = ret["candidate_action_type"]
        ret["loss_weights"] = torch.ones_like(ret["history_item_ids"]).reshape(1, -1)
        return ret


@dataclass
class RecoDataset:
    max_sequence_length: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset


def get_rank_dlrm_kr_dataloader(args, train_conf, model_conf, feature_conf, is_train):
    data_dir = args.train_data_dir if is_train else args.test_data_dir
    dataset = RankDLRMDataloader(
        padding_length=model_conf.get("max_sequence_length", 100),
        user_feat_config=feature_conf.get("user_feature_columns"),
        item_feat_config=feature_conf.get("item_feature_columns"),
        seq_feat_config=feature_conf.get("seq_feature_columns"),
        dataset_train_dir=args.train_data_dir,
        dataset_valid_dir=args.test_data_dir,
        rank=args.rank,
        world_size=args.world_size,
        is_train=is_train,
        file_format="csv",
        sep=",",
        exclude_features=[],
        num_rerank=1,
        infer_label_key="history_action_type"
    )
    logging.info(f'data is None {dataset is None}')
    return dataset


def get_reco_dataset(  # noqas
        args, train_conf, model_conf, feature_conf, dataloader_conf):
    if dataloader_conf["dataset_name"] == "kr-rank-dlrm":
        train_dataset = get_rank_dlrm_kr_dataloader(args, train_conf, model_conf, feature_conf, True)
        eval_dataset = get_rank_dlrm_kr_dataloader(args, train_conf, model_conf, feature_conf, False)

    all_item_ids = \
        [x + 1 for
         x in range(feature_conf.get('item_feature_columns')[feature_conf.get("infer_items_key")]["feature_count"]
                    )]

    return RecoDataset(
        max_sequence_length=model_conf.get("max_sequence_length", 100),
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def get_train_eval_dataloader(args, train_conf, model_conf, feature_conf, dataloader_conf):
    dataset = get_reco_dataset(
        args, train_conf, model_conf, feature_conf, dataloader_conf
    )

    train_data_loader = create_data_loader_ep_rank(
        dataset.train_dataset,
        feature_conf,
        train_conf,
        dataloader_conf,
        model_conf,
    )
    eval_data_loader = create_data_loader_ep_rank(
        dataset.eval_dataset,
        feature_conf,
        train_conf,
        dataloader_conf,
        model_conf,
    )

    return dataset, eval_data_loader, train_data_loader


def create_data_loader_ep(
        dataset: torch.utils.data.Dataset,
        feature_conf,
        train_conf,
        dataloader_conf,
        model_conf,
) -> torch.utils.data.DataLoader:
    item_feature_columns = feature_conf['item_feature_columns']
    itemid_column_name = feature_conf['infer_items_key']
    infer_ratings_key = feature_conf['infer_ratings_key']
    infer_timestamps_key = feature_conf['infer_timestamps_key']
    max_sequence_length = model_conf.get("max_sequence_length", 100)
    num_negatives = dataloader_conf['num_to_sample']
    training = dataset.is_train
    max_item_id = item_feature_columns[itemid_column_name]["feature_count"]
    all_item_ids = torch.arange(1, max_item_id + 1, dtype=torch.long)
    num_items = all_item_ids.size(0)
    dynamic_batch_conf = train_conf.get("use_dynamic_batch", {"enabled": False, "max_tokens": 100})
    is_dynamic_batch = dynamic_batch_conf.get("enabled", False)
    is_load_balance = train_conf.get("is_load_balance", False)

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        ds = worker_info.dataset
        ds.files = ds.files[worker_id::worker_info.num_workers]
        ds.init_multi_csv_iterator()

    batch_size = train_conf.get("train_batch_size", 256)
    if training and (is_dynamic_batch or is_load_balance):
        batch_size = 1
    elif not training:
        batch_size = int(os.environ.get("Batch_Size"))

    def collect_fn(data, feature_conf, model_conf, dataloader_conf, train_conf):
        from torchrec import KeyedJaggedTensor, JaggedTensor

        item_feature_columns = feature_conf['item_feature_columns']
        itemid_column_name, infer_ratings_key, infer_timestamps_key = \
            feature_conf['infer_items_key'], feature_conf['infer_ratings_key'], feature_conf['infer_timestamps_key']
        max_sequence_length = model_conf.get("max_sequence_length", 100)

        num_negatives = dataloader_conf['num_to_sample']
        is_dynamic_batch = train_conf.get("use_dynamic_batch", {"enabled": False}).get('enabled', False)
        max_item_id = item_feature_columns[itemid_column_name]["feature_count"]
        all_item_ids = torch.arange(1, max_item_id + 1, dtype=torch.long)
        num_items = all_item_ids.size(0)

        if training and (is_dynamic_batch or is_load_balance):
            data = data[0]

        bs, uids, past_lengths, tokennum, past_payloads = assemble_batch(data, infer_ratings_key,
                                                                         infer_timestamps_key, item_feature_columns,
                                                                         training)

        for key, lst in past_payloads.items():
            if len(lst) > 0 and isinstance(lst[0], torch.Tensor) and lst[0].dim() > 0:
                past_payloads[key] = torch.concat(lst)
            else:
                past_payloads[key] = torch.tensor(lst)
        past_ids = past_payloads.get(itemid_column_name).clone()

        if training:
            current_offset = 0
            jagged_parts = []
            past_ids_new = torch.zeros((bs, max_sequence_length), dtype=torch.int64)
            for i in range(bs):
                n = past_lengths[i]
                end = current_offset + n
                jagged_parts.append(past_ids[current_offset + 1:end])
                current_offset = end
                past_ids_new[i, :n] = 1
            past_payloads["positive_ids"] = torch.cat(jagged_parts) if jagged_parts else torch.empty(0,
                                                                                                     dtype=past_ids.dtype,
                                                                                                     device=past_ids.device)
            past_ids = past_ids_new
        else:
            past_ids = past_payloads.get(itemid_column_name).clone()
            past_payloads["positive_ids"] = past_ids

        def handle_negative_samples(past_lengths, output_shape, num_items, item_feature_id, training):
            def generate_negative_ids(output_shape, num_items, item_feature_id, all_item_ids):
                sampled_offsets = torch.randint(
                    low=0,
                    high=num_items,
                    size=output_shape,
                    dtype=item_feature_id.dtype,
                    device=item_feature_id.device,
                )
                negative_ids = all_item_ids.to(item_feature_id.device)[sampled_offsets.view(-1)].reshape(output_shape)
                return negative_ids

            bs, max_sequence_length = output_shape[0], output_shape[1] + 1
            if training:
                offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths)
                negative_ids = generate_negative_ids(output_shape, num_items, item_feature_id, all_item_ids)
                negative_ids = torch.ops.fbgemm.dense_to_jagged(negative_ids, [offsets])[0]
                negative_num = negative_ids.shape[1]
                negative_feature_id = negative_ids.reshape(-1)
                negative_lens = negative_feature_id.shape[0]
                lengths = torch.tensor(past_lengths)
                jagged_tensor = JaggedTensor(
                    values=item_feature_id.long(),
                    lengths=lengths
                )
                jagged_tensor_neg = JaggedTensor(
                    values=negative_feature_id.long(),
                    lengths=lengths * negative_num
                )
            else:
                negative_ids = generate_negative_ids(output_shape, num_items, item_feature_id, all_item_ids)
                negative_feature_id = negative_ids.reshape(-1)
                negative_lens = negative_feature_id.shape[0]
                item_ids_with_neg = torch.cat([
                    item_feature_id,
                    negative_feature_id
                ])
                item_lengths = [max_sequence_length] * bs
                lengths = torch.cat([torch.tensor(item_lengths), torch.tensor([negative_lens])])
                jagged_tensor = JaggedTensor(
                    values=item_ids_with_neg.long(),
                    lengths=lengths
                )
                jagged_tensor_neg = JaggedTensor(
                    values=torch.tensor([]).long(),
                    lengths=torch.zeros_like(lengths)
                )
            return negative_ids, jagged_tensor, jagged_tensor_neg

        item_feature_id = past_payloads.get(itemid_column_name) \
            if training else past_payloads.get(itemid_column_name).reshape(-1)
        output_shape = [bs, max_sequence_length - 1, num_negatives]
        negative_ids, negative_jagged_tensor, jagged_tensor_neg = handle_negative_samples(past_lengths, output_shape,
                                                                                          num_items,
                                                                                          item_feature_id, training)
        kjt_dict = {}
        kjt_dict[itemid_column_name] = negative_jagged_tensor
        kjt_dict[itemid_column_name + "_neg"] = jagged_tensor_neg
        item_kjt_with_neg = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        past_payloads["item_kjt_with_neg"] = item_kjt_with_neg
        past_payloads['negative_ids'] = negative_ids
        past_payloads['batch_size'] = torch.tensor(bs)

        features = SequentialFeatures(
            uid=uids,
            past_lengths=past_lengths,
            token_num=tokennum,
            batch_size=torch.LongTensor([bs]),
            past_ids=past_ids,
            past_embeddings=None,
            past_payloads=past_payloads
        )
        batch = Batch(features)
        return batch

    def collect_fn_c(data):
        return collect_fn(data, feature_conf, model_conf, dataloader_conf, train_conf)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collect_fn_c,
        shuffle=False,
        num_workers=dataloader_conf['num_workers'],
        prefetch_factor=dataloader_conf['prefetch_factor'],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True
    )

    return data_loader


def assemble_batch(data, infer_ratings_key, infer_timestamps_key, item_feature_columns, training, **kwargs):
    uids, past_lengths, past_payloads = [], [], {}
    bs = 0
    for row in data:
        bs += 1
        past_lengths.append(row["history_lengths"])
        uids.append(row['uid'])
        historical_ratings = row[infer_ratings_key]
        historical_timestamps = row[infer_timestamps_key]

        past_payloads.setdefault(infer_ratings_key, []).append(historical_ratings.unsqueeze(0))
        past_payloads.setdefault(infer_timestamps_key, []).append(historical_timestamps.unsqueeze(0))

        for feat_name, _ in item_feature_columns.items():
            feat_ids = row[feat_name]

            if training:
                past_payloads.setdefault(feat_name, []).append(feat_ids)
            else:
                past_payloads.setdefault(feat_name, []).append(feat_ids.unsqueeze(0))

        labels = row['labels']
        loss_weights = row['loss_weights']
        past_payloads.setdefault('loss_weights', []).append(loss_weights.unsqueeze(0))
        past_payloads.setdefault('labels', []).append(labels.unsqueeze(0))

    return bs, torch.LongTensor(uids), torch.LongTensor(past_lengths), \
        sum(past_lengths), past_payloads


def create_data_loader_ep_rank(
        dataset: torch.utils.data.Dataset,
        feature_conf,
        train_conf,
        dataloader_conf,
        model_conf,
) -> torch.utils.data.DataLoader:
    training = dataset.is_train
    dynamic_batch_conf = train_conf.get("use_dynamic_batch", {"enabled": False, "max_tokens": 100})
    is_dynamic_batch = dynamic_batch_conf.get("enabled", False)
    ec_feature_names = feature_conf["ec_feature_names"]

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        ds = worker_info.dataset
        ds.files = ds.files[worker_id::worker_info.num_workers]
        ds.init_multi_csv_iterator()

    # batch_size = train_conf.get("train_batch_size", 256)
    batch_size = int(os.environ.get("Batch_Size"))
    if training and is_dynamic_batch:
        batch_size = 1
    elif not training:
        # batch_size = train_conf.get("eval_batch_size", 256)
        batch_size = int(os.environ.get("Batch_Size"))

    def collect_fn(data, feature_conf, model_conf, dataloader_conf, train_conf):
        from torchrec import KeyedJaggedTensor, JaggedTensor

        item_feature_columns = feature_conf['seq_feature_columns']
        candidate_feature_columns = feature_conf['item_feature_columns']
        user_feature_columns = feature_conf['user_feature_columns']

        itemid_column_name, infer_ratings_key, infer_timestamps_key = \
            feature_conf['infer_items_key'], feature_conf['infer_ratings_key'], feature_conf['infer_timestamps_key']

        max_sequence_length = model_conf.get("max_sequence_length", 100)
        # max_sequence_length = int(os.environ.get("Non_Seq_Len"))

        is_dynamic_batch = train_conf.get("use_dynamic_batch", {"enabled": False}).get('enabled', False)

        if is_dynamic_batch:
            data = data[0]

        bs, uids, past_lengths, tokennum, past_payloads = assemble_batch_rank(data, infer_ratings_key,
                                                                              infer_timestamps_key,
                                                                              item_feature_columns,
                                                                              user_feature_columns,
                                                                              candidate_feature_columns, training)

        for key, lst in past_payloads.items():
            if len(lst) > 0 and isinstance(lst[0], torch.Tensor) and lst[0].dim() > 0:
                past_payloads[key] = torch.concat(lst)
            else:
                past_payloads[key] = torch.tensor(lst)

        past_ids = past_payloads.get(itemid_column_name).clone()

        if training:
            past_ids = torch.zeros((bs, max_sequence_length), dtype=torch.int64)
            for i in range(bs):
                n = past_lengths[i]
                past_ids[i, :n] = 1
        else:
            past_ids = past_payloads.get(itemid_column_name).clone()

        past_payloads['past_lengths'] = past_lengths
        features = SequentialFeatures(
            uid=uids,
            past_lengths=past_lengths,
            token_num=tokennum,
            batch_size=torch.LongTensor([bs]),
            past_ids=past_ids,
            past_embeddings=None,
            past_payloads=past_payloads
        )

        def prepare_batch(seq_features: SequentialFeatures):

            payloads = seq_features.past_payloads
            for ec_attr, feature_names in ec_feature_names.items():
                jt_dict = {}
                for feature_name in feature_names:
                    feature_id = payloads[feature_name]
                    batch_size, lengths = feature_id.size(0), feature_id.numel()

                    jt = JaggedTensor(
                        values=feature_id.reshape(-1).long(),
                        lengths=torch.tensor([lengths // batch_size] * batch_size, dtype=torch.int64)
                    )
                    jt_dict[feature_name] = jt
                payloads[ec_attr] = KeyedJaggedTensor.from_jt_dict(jt_dict)

            return Batch(seq_features)

        batch = prepare_batch(features)
        return batch

    def collect_fn_c(data):
        return collect_fn(data, feature_conf, model_conf, dataloader_conf, train_conf)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collect_fn_c,
        shuffle=False,
        num_workers=dataloader_conf['num_workers'],
        prefetch_factor=dataloader_conf['prefetch_factor'],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True
    )

    return data_loader


def assemble_batch_rank(data, infer_ratings_key, infer_timestamps_key, item_feature_columns, user_feature_columns,
                        candidate_feature_columns, training):
    uids, past_lengths, past_payloads = [], [], {}
    bs = 0
    candidate_timestamps_key = 'candidate_timestamps'
    for row in data:
        bs += 1
        past_lengths.append(row["history_lengths"])
        uids.append(row['uid'])
        historical_ratings = row[infer_ratings_key]
        historical_timestamps = row[infer_timestamps_key]
        candidate_timestamps = row[candidate_timestamps_key]

        past_payloads.setdefault(infer_ratings_key, []).append(historical_ratings.unsqueeze(0))
        past_payloads.setdefault(infer_timestamps_key, []).append(historical_timestamps.unsqueeze(0))
        past_payloads.setdefault(candidate_timestamps_key, []).append(candidate_timestamps.unsqueeze(0))

        for feat_name, _ in item_feature_columns.items():
            feat_ids = row[feat_name]
            past_payloads.setdefault(feat_name, []).append(feat_ids.unsqueeze(0))

        for feat_name, _ in candidate_feature_columns.items():
            feat_ids = row[feat_name]
            past_payloads.setdefault(feat_name, []).append(feat_ids.unsqueeze(0))

        for feat_name, _ in user_feature_columns.items():
            feat_ids = row[feat_name]
            past_payloads.setdefault(feat_name, []).append(feat_ids.unsqueeze(0))

        labels = row['labels']
        loss_weights = row['loss_weights']
        past_payloads.setdefault('loss_weights', []).append(loss_weights.unsqueeze(0))
        past_payloads.setdefault('labels', []).append(labels.unsqueeze(0))

    return bs, torch.LongTensor(uids), torch.LongTensor(past_lengths), \
        torch.LongTensor([sum(past_lengths)]), past_payloads


class SequentialFeatures:
    def __init__(self, uid: torch.Tensor, past_lengths: torch.Tensor,
                 past_ids: torch.Tensor, token_num: int, batch_size: torch.Tensor,
                 past_payloads: Dict[str, torch.Tensor], past_embeddings: Optional[torch.Tensor] = None) -> None:
        self.uid = self._assert_not_none(uid, "uid cannot be None")
        self.past_lengths = self._assert_not_none(past_lengths, "past_lengths cannot be None")
        self.past_ids = self._assert_not_none(past_ids, "past_ids cannot be None")
        self.past_payloads = self._assert_not_none(past_payloads, "past_payloads cannot be None")
        self.token_num = self._assert_not_none(token_num, "token_num cannot be None")
        self.batch_size = self._assert_not_none(batch_size, "batch_size cannot be None")
        self.past_embeddings = past_embeddings
        self.offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths)
        self.loss_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths - 1)
        self.offsets_list = self.offsets.tolist()
        self.loss_offsets_list = self.loss_offsets.tolist()

    @staticmethod
    def _assert_not_none(obj, errmsg):
        if obj is None:
            raise ValueError(errmsg)
        return obj

    def to(self, device: torch.device, non_blocking: bool = False) -> "SequentialFeatures":
        self.uid = self.uid.to(device, non_blocking=non_blocking)
        self.past_lengths = self.past_lengths.to(device, non_blocking=non_blocking)
        self.past_ids = self.past_ids.to(device, non_blocking=non_blocking)
        self.offsets = self.offsets.to(device, non_blocking=non_blocking)
        self.loss_offsets = self.loss_offsets.to(device, non_blocking=non_blocking)
        self.batch_size = self.batch_size.to(device, non_blocking=non_blocking)
        for k in self.past_payloads.keys():
            self.past_payloads[k] = self.past_payloads[k].to(device, non_blocking=non_blocking)
        if self.past_embeddings is not None:
            self.past_embeddings = self.past_embeddings.to(device, non_blocking=non_blocking)
        return self

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self.uid.record_stream(stream)
        self.past_lengths.record_stream(stream)
        self.past_ids.record_stream(stream)
        self.batch_size.record_stream(stream)
        for k in self.past_payloads.keys():
            self.past_payloads[k].record_stream(stream)
        if self.past_embeddings is not None:
            self.past_embeddings.record_stream(stream)

    def pin_memory(self) -> "SequentialFeatures":
        self.uid = self.uid.pin_memory()
        self.past_lengths = self.past_lengths.pin_memory()
        self.past_ids = self.past_ids.pin_memory()
        self.batch_size = self.batch_size.pin_memory()
        for k in self.past_payloads.keys():
            self.past_payloads[k] = self.past_payloads[k].pin_memory()
        if self.past_embeddings is not None:
            self.past_embeddings = self.past_embeddings.pin_memory()
        return self

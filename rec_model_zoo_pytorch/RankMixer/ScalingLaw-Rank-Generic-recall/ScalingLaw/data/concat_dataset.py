import ast
import logging
import os
from typing import Dict, List, Optional, Tuple
import random
import pandas as pd
import numpy as np
import torch
import traceback
from torch.utils.data import IterableDataset

class MultiCSVIterator:
    """用于读取csv文件的迭代器。"""
    
    def __init__(self, file_paths:List[str], sep:str, rank: int, world_size: int, column_names, inner_delim,
                 itemid_column_name='item_id', file_format='csv', split_chunks_by_worker_id=False, chunksize=10000000,
                 use_column_names=False, use_cols = None, shuffle=False, epoch_num=0) -> None:
        """
        初始化MultiCSVIterator。

        :param file_paths: csv文件路径列表。
        :param sep: csv文件的分隔符。
        :param rank: 用于分布式训练的rank，0表示主进程。
        :param world_size: 分布式并行总进程数。
        :param column_names: csv文件的列名。
        :param inner_delim: 内部分隔符，用于处理列中的多个值。
        :param itemid_column_name: 表示物品ID的列名，默认为'item_id'。
        :param use_column_names: 表示物品ID的列名，默认为'item_id'。
        """

        self.file_paths = file_paths
        self.sep = sep
        self.rank = rank
        self.world_size = world_size

        self.current_file_index = 0
        self.chunk_iterator = iter([])
        self.chunk_data = None
        self.total_read_count = 0

        self.column_names = column_names
        self.inner_delim = inner_delim

        self.itemid_column_name = itemid_column_name
        self.worker_info = torch.utils.data.get_worker_info()
        self.file_format = file_format
        self.split_chunks_by_worker_id = split_chunks_by_worker_id
        self.shuffle = shuffle
        self.increment = 1 if self.split_chunks_by_worker_id else self.world_size
        self.current_data_index = 0 if self.split_chunks_by_worker_id else self.rank
        self.chunk_size = chunksize
        self.use_column_names = use_column_names

        self.current_file = None
        self.current_chunk = None 
        self.valid_data_size = 0
        self.use_cols = use_cols
        self.epoch_num=epoch_num
    
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
            elif self.file_format == 'parquet':
                import pyarrow.parquet as pq
                self.current_file = pq.ParquetFile(self.file_paths[self.current_file_index]).iter_batches(
                    self.chunk_size)
            self.current_chunk_idx = 0
        
        if self.current_chunk is None:
            try:
                # remove first row if is text information
                if self.file_format == 'csv':
                    self.current_chunk = self.current_file.__next__()
                elif self.file_format == 'parquet':
                    self.current_chunk = self.current_file.__next__().to_pandas()

                self.valid_data_size = len(self.current_chunk) - len(self.current_chunk) % self.world_size
                if self.shuffle:
                    self.shuffled_indices = [i for i in range(self.valid_data_size)]
                    random.seed(self.epoch_num)
                    random.shuffle(self.shuffled_indices)
                    #logging.info(f'print debug self.shuffled_indices || rank {self.rank} || self.epoch_num {self.epoch_num} || indices: {self.shuffled_indices[:20]}')
            except StopIteration:
                self.current_data_index = self.rank
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
            self.current_data_index = 0
            self.current_chunk = None
            return self.__next__()

class RecallDataloader(IterableDataset):
    """用于处理按时间逆序排列的数据集。"""

    def __init__(
            self,
            ratings_file: str,
            padding_length: int,
            ignore_last_n: int,
            shift_id_by: int = 0,
            chronological: bool = True,
            sample_ratio: float = 1.0,
            sep: str = ',',
            rank: int = 0,
            world_size: int = 1,
            seq_columns=None,
            nonseq_columns=None,
            itemid_column_name='item_id',
            ratings_column_name='event_id',
            timestamps_column_name='time_stamp',
            userid_column_name='device_id_sha256',
            inner_delim: str = '^',
            is_train: bool = True,
            cut_off_time: int = 1727625600,
            num_rerank=200,
            use_repadding: bool = False,
            cut_off_train_data: bool = False,
            token_per_item: int = 2,
            user_id_hex: int = 1,
            shuffle: bool = False
    ) -> None:
        super().__init__()
        """
        初始化DatasetV8。
        :param ratings_file: 文件路径。
        :param padding_length: 填充长度。
        :param ignore_last_n: 忽略最后n个数据项。
        :param shift_id_by: ID偏移量。
        :param chronological: 默认为True, 处理后按时间正序排列。
        :param sample_ratio: 采样比例。
        :param sep: 分隔符。
        :param rank: 用于分布式训练的rank。
        :param world_size: 用于分布式训练的世界大小。
        :param seq_columns: 序列列名，这里指货/场特征名。
        :param nonseq_columns: 非序列列名，这里指人特征名。
        :param itemid_column_name: 物品ID列名。
        :param ratings_column_name: 评分列名。
        :param timestamps_column_name: 时间戳列名。
        :param inner_delim: 内部分隔符。
        :param is_train: 是否为训练数据。
        :param cut_off_time: 训练/验证切分时间戳, 默认为093000对应的时间戳。
        """
        
        if seq_columns is None:
            raise ValueError('seq_columns should not be None')
        if nonseq_columns is None:
            raise ValueError('nonseq_columns should not be None')

        self.rank = rank
        self.world_size = world_size

        try:
            files = os.listdir(ratings_file)
        except FileNotFoundError as fnf_error:
            logging.error('No such file or directory: %s', ratings_file)
            raise fnf_error
        except NotADirectoryError as na_dir_error:
            logging.error('Not a directory: %s', ratings_file)
            raise na_dir_error
        except OSError as os_error:
            logging.error('OS error: %s', ratings_file)
            raise os_error

        self.files = [ratings_file + '/' + f for f in files if f.endswith('.csv')]

        self.current_ratings_frame_len = 0
        self.current_ratings_frame_idx = 0
        self.data_idx = 0
        self.ratings_frame = None

        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio

        self.seq_columns = seq_columns
        self.nonseq_columns = nonseq_columns
        self.seq_column_names = [k for k, _ in seq_columns.items()]
        self.nonseq_column_names = [k for k, _ in nonseq_columns.items()]
        self.itemid_column_name = itemid_column_name
        self.ratings_column_name = ratings_column_name
        self.timestamps_column_name = timestamps_column_name
        self.userid_column_name = userid_column_name
        self.column_names = self.seq_column_names + self.nonseq_column_names + \
                            [ratings_column_name, timestamps_column_name]

        self.inner_delim = inner_delim
        self.cut_off_time = cut_off_time
        self.is_train = is_train
        self.cut_off_train_data = cut_off_train_data

        self.sep = sep
        self.multi_csv_iterator = None
        self._max_candidate_num: int = num_rerank

        self.use_repadding = use_repadding
        if self.use_repadding and rank == 0:
            logging.info('Using sequence expand by repadding')
        self.token_per_item = token_per_item
        self.user_id_hex= user_id_hex
        self.debug_print = False
        self.shuffle = shuffle
        self.epoch_num = 0

    def init_multi_csv_iterator(self) -> None:
        if not self.multi_csv_iterator:
            worker_info = torch.utils.data.get_worker_info()
            self.multi_csv_iterator = MultiCSVIterator(self.files, self.sep, self.rank, self.world_size,
                                                       self.column_names,
                                                       self.inner_delim, itemid_column_name=self.itemid_column_name,
                                                       split_chunks_by_worker_id=False,shuffle=self.shuffle,
                                                       epoch_num=self.epoch_num)

    def set_epoch_num(self, epoch_num : int):
        self.epoch_num = epoch_num
        
    def __iter__(self):
        # 当前df的数据索引
        it = map(self.load_item, self.multi_csv_iterator)
        return it

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        """
        加载单个数据项。

        :param data: 单个数据项。
        :return: 处理后的数据项，包含 history_lengths, test_position, ratings, loss_weights, labels, timestamps
        """

        def eval_as_list(x: str, ignore_last_n) -> List[int]:

            try:
                if not isinstance(x, str):
                    x = str(x)

                x = x.replace(self.inner_delim, ',')
                # 方法1：使用封装的ast方法
                y = ast.literal_eval(x)
                if isinstance(y, float):
                    y = int(y)
                y_list = [y] if isinstance(y, int) else list(y)
                if not self._chronological:
                    y_list.reverse()
                if ignore_last_n > 0:
                    # for training data creation
                    y_list = y_list[:-ignore_last_n]
                return y_list

            except Exception as e:
                print("x.type: ", type(x))
                print("x: ", x)
                traceback.print_exc()
                return None

        def eval_int_list(x, ignore_last_n: int, shift_id_by: int, sampling_kept_mask: Optional[List[bool]]) -> Tuple[
            List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        self._sample_ratio = 1.0
        if self._sample_ratio < 1.0:
            labels = eval_as_list(data[self.ratings_column_name], self._ignore_last_n)
        else:
            sampling_kept_mask = None

        non_seq_feature = {}
        seq_feature_list = {}
        seq_feature_len = {}
        candidate_feature_list = {}

        for column_name in self.seq_column_names:
            seq_feature, seq_history_len = eval_int_list(
                data[column_name], 0 if self.cut_off_train_data else self._ignore_last_n, 
                shift_id_by=self._shift_id_by, sampling_kept_mask=sampling_kept_mask
            )
            seq_feature_list[column_name] = seq_feature
            seq_feature_len[column_name] = seq_history_len

        for column_name in self.nonseq_column_names:
            feat_list = eval_as_list(data[column_name], 0)
            feat = feat_list[0] if len(feat_list) > 0 else 0
            non_seq_feature[column_name] = feat

        ratings, ratings_len = eval_int_list(
            data[self.ratings_column_name], 0 if self.cut_off_train_data else self._ignore_last_n,
            0, sampling_kept_mask=sampling_kept_mask
        )
        timestamps, timestamps_len = eval_int_list(
            data[self.timestamps_column_name], 0 if self.cut_off_train_data else self._ignore_last_n,
            0, sampling_kept_mask=sampling_kept_mask
        )

        if timestamps_len != ratings_len:
            raise ValueError("timestamps len %s differs from ratings len %s.", (timestamps_len, ratings_len))

        def _truncate_or_pad_seq(y: List[int], target_len: int) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                y = y[-target_len:]
            if len(y) != target_len:
                raise ValueError
            return y

        candidate_ratings, candidate_timestamps = [], [] # deprecated for recall

        timestamps_original = [t for t in timestamps]

        ratings.reverse()
        timestamps.reverse()
        timestamps_original.reverse()
        candidate_timestamps.reverse()
        candidate_ratings.reverse()

        max_seq_len = self._padding_length
        max_candidate_num = self._max_candidate_num
        original_seq_len = len(timestamps)

        if self.cut_off_train_data:
            split_pos = min([i if ts > self.cut_off_time else original_seq_len for i, ts in enumerate(timestamps)]) - self._ignore_last_n
            ratings = ratings[:split_pos]
            timestamps = timestamps[:split_pos]
        
        history_length = min(len(timestamps), max_seq_len)
        ratings = _truncate_or_pad_seq(ratings, max_seq_len)
        timestamps = _truncate_or_pad_seq(timestamps, max_seq_len)

        actual_candidate_num = min(len(candidate_timestamps), max_candidate_num)

        def candidate_feature_interleave(candidate_feature: list, n: int = 2) -> list:
            length = len(candidate_feature)
            return [candidate_feature[i // n] for i in range(length * n)]


        if not self.is_train:
            candidate_ratings = _truncate_or_pad_seq(candidate_ratings, max_candidate_num)
            candidate_timestamps = _truncate_or_pad_seq(candidate_timestamps, max_candidate_num)

            candidate_ratings = candidate_feature_interleave(candidate_ratings, self.token_per_item)
            candidate_timestamps = candidate_timestamps * self.token_per_item

        for k, v in seq_feature_list.items():
            candidate_v = []
            v.reverse()
            candidate_v.reverse()
            if self.cut_off_train_data:
                v = v[:split_pos]
            v = _truncate_or_pad_seq(v, max_seq_len)
            seq_feature_list[k] = v

            if not self.is_train:
                candidate_v = _truncate_or_pad_seq(candidate_v, max_candidate_num)
                candidate_feature_list['candidate_' + k] = candidate_feature_interleave(candidate_v, self.token_per_item)

        # 处理训练数据标签
        labels = ratings

        # 处理模型权重
        loss_weights = [1 for _ in ratings]

        # 间隔处loss weights为0，但是rating需要在embedding内
        if self.use_repadding:
            ratings = [0 if x == -1 else x for x in ratings]

        # aid为64位16进制数字，为了能让torch.int64不溢出，取其前15位
        def hex_str_to_int_list(hex_str):
            int_list = []
            for i in range(len(hex_str) // 15 + 1):
                result = int(hex_str[15*i:15*(i+1)], 16)
                int_list.append(result)
            return int_list

        user_id = str(data[self.userid_column_name])
        if self.user_id_hex:
            uid = hex_str_to_int_list(user_id)
        else:
            uid = int(user_id)
            
        if self.debug_print:
            if self._ignore_last_n + split_pos >= len(timestamps_original):
                logging.info(f"full  feature list : {seq_feature_list[self.itemid_column_name]} || timestamps {timestamps} || original_timestamps{timestamps_original} ||cut_off_time {self.cut_off_time}")
            else:
                logging.info(f'debug timestamps original {timestamps_original}')
                logging.info(f'debug split_pos {split_pos} || critical time {timestamps_original[split_pos+self._ignore_last_n]} || item_id list {seq_feature_list[self.itemid_column_name]} || debug timestamp : \n {timestamps}')
                self.debug_print = False
        
        ret = {
            "uid": torch.tensor(uid, dtype=torch.int64),
            "history_lengths": history_length,
            'ratings': torch.tensor(ratings, dtype=torch.int64),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            'timestamps': torch.tensor(timestamps, dtype=torch.int64),
        }

        for k, v in non_seq_feature.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)

        for k, v in seq_feature_list.items():
            ret[k] = torch.tensor(v, dtype=torch.int64)
        
        if not self.is_train:
            candidate_labels = candidate_ratings
            for k, v in candidate_feature_list.items():
                ret[k] = torch.tensor(v, dtype=torch.int64)
            ret["candidate_item_id"] = ret.get("candidate_" + self.itemid_column_name, torch.tensor([0]))
            ret["actual_num_rerank"] = actual_candidate_num
            ret["candidate_labels"] = torch.tensor(candidate_labels, dtype=torch.int64)
            ret["candidate_timestamps"] = torch.tensor(candidate_timestamps, dtype=torch.int64)

        return ret
    

def negative_sampling_mask(data1_supervise, data2_label, rate=0.07):
    """
    根据data1和data2生成掩码，并根据指定的采样率进行负采样。
    1. 目标榜单的训测数据
    2. 首先属于目标榜单（暂略）
    3. 然后supervise_list = 1,
    4. 然后负样本 0.07采样路采样

    :param data1_supervise: 标识是否用于监督训练的数据（0为否，1为是）
    :param data2_label: 用户的行为数据（0为负样本，非0为正样本）
    :param rate: 负采样的比率，默认是0.07。
    :return: 生成的掩码数组。
    """
    data1_supervise = np.array(data1_supervise)
    data2_label = np.array(data2_label)
    assert len(data1_supervise) == len(data2_label), "data1和data2的长度必须相等"
    N = len(data1_supervise)

    # 初始化掩码为全1（True）
    mask = np.ones(N, dtype=bool)

    # 按照要求设置部分mask值为0（False）
    mask[(data1_supervise == 1) & (data2_label == 0)] = False

    # 获取需要进行负采样的索引列表
    false_indices = np.where(mask == False)[0]

    # 计算采样数量
    sample_num = round(len(false_indices) * rate)

    # 确保至少采样一个
    if sample_num >= 1:
        # 随机选择索引进行采样
        sampled_indices = np.random.choice(false_indices, size=sample_num, replace=False)

        # 更新掩码：将所有 false_indices 设为 False，然后只将 sampled_indices 设为 True
        mask.fill(True)
        mask[false_indices] = False
        mask[sampled_indices] = True

    else:
        # 如果没有符合条件的索引，则保持初始掩码不变或根据需求调整
        mask = None
#         print("没有找到符合条件的索引进行负采样。")

    return mask



        
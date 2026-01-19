import os
from dataclasses import dataclass
from typing import List
import logging

import torch

from data.concat_dataset import RecallDataloader
from modeling.generic.utils.constants import Const


@dataclass
class RecoDataset:
    """
    数据集类，用于存储训练和评估数据集的相关信息。
    """
    max_sequence_length: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset


def get_format_csv(data_dir, pth) -> str:
    return os.path.join(data_dir, pth)


def get_reco_dataset(
        dataset: str,
        data_dir: str,
        max_sequence_length: int,
        rank: int,
        world_size: int,
        pth: str,
        chronological: bool = True,
        feature_conf: dict = None,
        num_rerank=256,
        cut_off_time: int = None,        
        token_per_item: int = 2
) -> RecoDataset:
    """
    创建并返回一个RecoDataset对象, 包含训练和评估数据集。
    
    :param data_dir: 数据目录。
    :param max_sequence_length: 序列的最大长度。
    :param rank: 全局rank ID。
    :param world_size: 分布式并行总进程数。
    :param pth: 文件路径。
    :param chronological: 是否按时间正序排列。
    :param feature_conf: 特征配置字典。
    :return: RecoDataset对象, 包含训练和评估数据集的相关信息。
    """
    if dataset == "ag-recall":
        cut_off_time = feature_conf.get("cut_off_time", cut_off_time)
        logging.info(f'print cut off time : {cut_off_time}')
        train_dataset = RecallDataloader(
            ratings_file=get_format_csv(data_dir, pth),
            padding_length=max_sequence_length,
            ignore_last_n=1,
            chronological=False,
            sep=';',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            userid_column_name=feature_conf.get('raw_userid_column'),
            itemid_column_name=feature_conf.get('raw_itemid_column'),
            ratings_column_name=feature_conf.get('raw_ratings_column'),
            timestamps_column_name=feature_conf.get('raw_timestamps_column'),
            is_train=True,
            cut_off_time=cut_off_time,
            cut_off_train_data=True,
        )
        eval_dataset = RecallDataloader(
            ratings_file=get_format_csv(data_dir, pth),
            padding_length=max_sequence_length,
            ignore_last_n=0,
            chronological=False,
            sep=';',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            userid_column_name=feature_conf.get('raw_userid_column'),
            itemid_column_name=feature_conf.get('raw_itemid_column'),
            ratings_column_name=feature_conf.get('raw_ratings_column'),
            timestamps_column_name=feature_conf.get('raw_timestamps_column'),
            is_train=False,
            cut_off_time=cut_off_time,
            num_rerank=0,
            cut_off_train_data=True,
        )
    elif dataset == "amazon-recall":
        cut_off_time = feature_conf.get("cut_off_time", cut_off_time)
        train_dataset = RecallDataloader(
            ratings_file=get_format_csv(data_dir, pth),
            padding_length=max_sequence_length,
            ignore_last_n=1,
            chronological=True,
            sep=',',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            userid_column_name=feature_conf.get('raw_userid_column'),
            itemid_column_name=feature_conf.get('raw_itemid_column'),
            ratings_column_name=feature_conf.get('raw_ratings_column'),
            timestamps_column_name=feature_conf.get('raw_timestamps_column'),
            is_train=True,
            cut_off_time=cut_off_time,
            cut_off_train_data=False,
            token_per_item=feature_conf.get('token_per_item'),
            user_id_hex = 0,
            shuffle=True,
            shift_id_by=1
        )
        eval_dataset = RecallDataloader(
            ratings_file=get_format_csv(data_dir, pth),
            padding_length=max_sequence_length,
            ignore_last_n=0,
            chronological=True,
            sep=',',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            userid_column_name=feature_conf.get('raw_userid_column'),
            itemid_column_name=feature_conf.get('raw_itemid_column'),
            ratings_column_name=feature_conf.get('raw_ratings_column'),
            timestamps_column_name=feature_conf.get('raw_timestamps_column'),
            is_train=False,
            cut_off_time=cut_off_time,
            num_rerank=0,
            cut_off_train_data=False,
            token_per_item=feature_conf.get('token_per_item'),
            user_id_hex = 0,
            shuffle=True,
            shift_id_by=1
        )
    elif dataset == "movielens-1m":
        cut_off_time = feature_conf.get("cut_off_time", cut_off_time)
        train_dataset = RecallDataloader(
            ratings_file=get_format_csv(data_dir, pth),
            padding_length=max_sequence_length,
            ignore_last_n=1,
            chronological=True,
            sep=',',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            userid_column_name=feature_conf.get('raw_userid_column'),
            itemid_column_name=feature_conf.get('raw_itemid_column'),
            ratings_column_name=feature_conf.get('raw_ratings_column'),
            timestamps_column_name=feature_conf.get('raw_timestamps_column'),
            is_train=True,
            cut_off_time=cut_off_time,
            cut_off_train_data=False,
            token_per_item=feature_conf.get('token_per_item'),
            user_id_hex = 0,
            shuffle=True,
        )
        eval_dataset = RecallDataloader(
            ratings_file=get_format_csv(data_dir, pth),
            padding_length=max_sequence_length,
            ignore_last_n=0,
            chronological=True,
            sep=',',
            rank=rank,
            world_size=world_size,
            seq_columns=feature_conf.get('item_feature_columns'),
            nonseq_columns=feature_conf.get('user_feature_columns'),
            userid_column_name=feature_conf.get('raw_userid_column'),
            itemid_column_name=feature_conf.get('raw_itemid_column'),
            ratings_column_name=feature_conf.get('raw_ratings_column'),
            timestamps_column_name=feature_conf.get('raw_timestamps_column'),
            is_train=False,
            cut_off_time=cut_off_time,
            num_rerank=0,
            cut_off_train_data=False,
            token_per_item=feature_conf.get('token_per_item'),
            user_id_hex = 0,
            shuffle=True,
        )
        
    all_item_ids = [x + 1 for x in range(feature_conf.get('item_feature_columns')[feature_conf.get("infer_items_key")]["feature_count"])]
    #TODO: fix filtered item ids
    if dataset == "movielens-1m":
        diff_all_item_ids = [91, 221, 323, 622, 646, 677, 686, 689, 740, 817, 883, 995, 1048, 1072, 1074, 1182, 1195, 1229, 1239, 1338, 1402, 1403, 1418, 1435, 1451, 1452, 1469, 1478, 1481, 1491, 1492, 1505, 1506, 1512, 1521, 1530, 1536, 1540, 1560, 1576, 1607, 1618, 1634, 1637, 1638, 1691, 1700, 1712, 1736, 1737, 1745, 1751, 1761, 1763, 1766, 1775, 1778, 1786, 1790, 1800, 1802, 1803, 1808, 1813, 1818, 1823, 1828, 1838, 3815]
        all_item_ids = [item for item in all_item_ids if item not in set(diff_all_item_ids)]
    # logging.info(f'dataloader print all item ids {all_item_ids}')
    
    return RecoDataset(
        max_sequence_length=max_sequence_length,
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

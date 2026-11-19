
import os
import sys
import time

import json, yaml
import pandas as pd

from datetime import datetime, timedelta
from typing import Dict
from modeling.generic.utils.constants import Const

import logging

import torch
import collections
from typing import Any
from torch.fx import Interpreter
from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
def read_yml(file_path, use_e=False):
    """
    :param file_path: 文件路径
    :param use_e: yml文件中是否含有科学计数法
    :return:
    """
    with open(file_path, 'r', encoding='UTF-8') as yml_file:
        result = yaml.safe_load(yml_file)
    return result


def read_json(json_path):
    """
    读取json格式文件
    :param json_path: 文件路径
    :return:
    """
    with open(json_path, "r", encoding='utf-8') as f:
        result = json.loads(f.read())
    return result


def get_config(file_path):
    """
    :param file_path: 文件路径
    :return config_dict: 配置文件字典
    """
    # 读取配置文件
    config_dict = read_yml(file_path, use_e=True)

    return config_dict


def weird_division(x, y):
    """
    :param x: 被除数
    :param y: 除数
    :return x/y: 商
    """
    if y < Const.EPS and y >= 0:
        y = Const.EPS
    elif y > -Const.EPS and y < 0:
        y = -Const.EPS
    return x / y


def refine_feat_and_model_conf(dataset: str, 
                               feature_conf: Dict, 
                               model_conf: Dict,
                               feature_map_dir_or_path, 
                               model_cfg, **kwargs):
    """
    1. 设置cut_off_time
    2. 从featuremap中读取每个特征的最大值, 写入feature_conf中每个用户/商品特征的feature_count字段.
    3. 根据feature_conf里positive_behavior字段指定的正向行为，配置行为到标签的映射和行为总数。

    """
    if dataset == "music-scalingraw-rank":
        # 设置cut_off_time
        period = feature_conf.get("period")
        date_format = "%Y%m%d-%H%M%S"
        date_obj = datetime.strptime(period, date_format)
        cut_off_date_obj = date_obj - timedelta(days=int(feature_conf["cut_off_bias"]))
        cut_off_date_obj = cut_off_date_obj.replace(hour=0, minute=0, second=0, microsecond=0)

        logging.info("cut off time is set to %s", cut_off_date_obj.strftime(date_format))

        cut_off_timestamp = int(cut_off_date_obj.timestamp())
        feature_conf["cut_off_time"] = cut_off_timestamp
        # 耗时较长，improve?
        feature_map_files = os.listdir(feature_map_dir_or_path)
        all_feature_map_df = []
        for feature_map_file in feature_map_files:
            feature_map_df = pd.read_orc(os.path.join(feature_map_dir_or_path, feature_map_file))
            all_feature_map_df.append(feature_map_df)
        all_feature_map_df = pd.concat(all_feature_map_df, ignore_index=True)

        if feature_conf["get_count_from_featuremap"]:
            prefix_to_input_key_convert = feature_conf["prefix_to_input_key_converter"]
            user_feature_prefix_list = feature_conf["feature_map_user_feature_prefix"]
            for prefix in user_feature_prefix_list:
                if prefix not in prefix_to_input_key_convert:
                    logging.warning("Key %s is not in prefix_to_input_key_converter, check your config!"
                                    "Using the original prefix instead.", prefix)
                    feature_name = prefix
                else:
                    feature_name = prefix_to_input_key_convert[prefix]
                assoicated = feature_conf["user_feature_columns"][feature_name].get("associated", None)
                if assoicated is not None:
                    logging.info("Feature %s is associated with %s, skip.", feature_name, assoicated)
                    continue
                feature_dtype = feature_conf["user_feature_columns"][feature_name].get("dtype", None)
                if feature_dtype == "con" or feature_dtype == "pref":
                    logging.info("Feature %s is continuous or pref, skip.", feature_name)
                    continue
                user_feature_df = all_feature_map_df[all_feature_map_df["feature_name"].str.contains(prefix)]
                if len(user_feature_df) == 0:
                    raise ValueError("In the feature map, there is no feature named %s, check your config!" % prefix)
                feature_count = user_feature_df["feature_id"].astype(int).max()
                if feature_name not in feature_conf["user_feature_columns"]:
                    logging.warning("%s is not in user_feature_columns, check your config!" % feature_name)
                    feature_conf["user_feature_columns"][feature_name] = {}
                feature_conf["user_feature_columns"][feature_name]["feature_count"] = feature_count + 1
                logging.info('The maximum ID of feature %s has been set to %s.', feature_name, feature_count + 1)
            
            item_feature_prefix_list = feature_conf["feature_map_item_feature_prefix"]
            for prefix in item_feature_prefix_list:
                if prefix not in prefix_to_input_key_convert:
                    logging.warning("Key %s is not in prefix_to_input_key_converter, check your config!"
                                    "Using the original prefix instead." % prefix)
                    feature_name = prefix
                else:
                    feature_name = prefix_to_input_key_convert[prefix]
                assoicated = feature_conf["item_feature_columns"][feature_name].get("associated", None)
                if assoicated is not None:
                    logging.info("Feature %s is associated with %s, skip.", feature_name, assoicated)
                    continue
                feature_dtype = feature_conf["item_feature_columns"][feature_name].get("dtype", None)
                if feature_dtype == "con" or feature_dtype == "pref":
                    logging.info("Feature %s is continuous or list, skip.", feature_name)
                    continue
                item_feature_df = all_feature_map_df[all_feature_map_df["feature_name"].str.contains(prefix)]
                if len(item_feature_df) == 0:
                    raise ValueError("In the feature map, there is no feature named %s, check your config!" % prefix)
                feature_count = item_feature_df["feature_id"].astype(int).max()

                if feature_name not in feature_conf["item_feature_columns"]:
                    logging.warning("%s is not in item_feature_columns, check your config!" % feature_name)
                    feature_conf["item_feature_columns"][feature_name] = {}
                feature_conf["item_feature_columns"][feature_name]["feature_count"] = feature_count + 1
                logging.info('The maximum ID of feature %s has been set to %s.', feature_name, feature_count + 1)

        # 处理行为评分
        positive_behavior = feature_conf.get("positive_behavior",
                                            ["behavior_collect", "behavior_download", "behavior_play"])
        behavior_prefix = feature_conf.get("behavior_prefix")
        behavior_df = all_feature_map_df[all_feature_map_df["feature_name"].str.contains(behavior_prefix)]
        label_dict = {}
        for behavior_row in behavior_df.itertuples(index=False):
            behavior_name, behavior_id = behavior_row[0], behavior_row[1]
            if behavior_name in positive_behavior:
                label_dict[int(behavior_id)] = 1
            else:
                label_dict[int(behavior_id)] = 0
        feature_conf['behavior_map'] = label_dict
        logging.info('The behavior map is %s.', feature_conf['behavior_map'])
        model_conf['num_ratings'] = len(label_dict)
        feature_map = all_feature_map_df.to_dict()
        feature_map = {feature_map["feature_name"][i] : feature_map["feature_id"][i] for i in feature_map["feature_name"]}
        feature_map_keys = list(feature_map.keys())
        # feature map里不应该包含user_id，因为数量太多会导致保存下来的feature map过大。这里检查一下，如果有则删去
        removed = []
        for k in feature_map_keys:
            if "user_id" in k:
                feature_map.pop(k, None)
                removed.append(k)
        logging.info("removed %s keys from the original feature map", len(removed))
    elif dataset == "ads":
        feature_map = read_json(feature_map_dir_or_path)
        # put max index into feature columns
        _feature_conf = [feature_conf["item_feature_columns"], feature_conf["user_feature_columns"]] 
        for f in _feature_conf:
            for col_name, col_info in f.items():
                col_info['feature_count'] = feature_map['maxIndexMap'][col_name]
        date = feature_conf['date']
        # 'data_delay_1d': 数据包含当前周期全天数据，测试集即从当前周期下一天开始向前计算，默认为True
        cut_off_time = datetime.strptime(date, "%Y%m%d") - timedelta(hours=int(feature_conf['test_hours']))
        if feature_conf['data_delay_1d']:
            cut_off_time += timedelta(days=1)
        cut_off_time = int(time.mktime(cut_off_time.timetuple()))
        logging.info(f'cutoff time for evaluation is set to {cut_off_time}')
        feature_conf['cut_off_time'] = cut_off_time
        feature_map = feature_map['sparse']
    elif dataset == "ag-rank" or dataset == "ag-recall":
        feature_map = read_json(feature_map_dir_or_path)
        feature_conf, model_conf = feature_conf, model_conf

        period = feature_conf.get("period")
        date_format = "%Y%m%d-%H%M%S"
        date_obj = datetime.strptime(period, date_format)
        cut_off_date_obj = date_obj - timedelta(days=int(feature_conf["cut_off_bias"]))
        cut_off_date_obj = cut_off_date_obj.replace(hour=0, minute=0, second=0, microsecond=0)

        logging.info("cut off time is set to %s", cut_off_date_obj.strftime(date_format))

        cut_off_timestamp = int(cut_off_date_obj.timestamp())
        feature_conf["cut_off_time"] = cut_off_timestamp
    elif dataset == "ads-recall" or dataset == "foundation-pretrain" or dataset == "amazon-recall" or dataset == "movielens-1m":
        feature_map={}
        feature_conf["cut_off_time"] = 0
    else:
        raise NotImplementedError("Unknown dataset")

    # 当配置使用GRModelEp模型时(embedding parallel，当前使用torchrec实现)时，检查依赖的算子是否安装，否则fallback回GR_model
    if model_cfg[Const.MODULE_NAME] == "GRModelEp":
        from modeling.generic.sequential_v2 import HAS_TORCHREC
        if not HAS_TORCHREC:
            logging.warning("Required ops of torchrec not installed, fallback model to 'GR_model'")
            model_cfg[Const.MODULE_NAME ] = "GR_model"
    model_conf["root_model_type"] = model_cfg[Const.MODULE_NAME]
    return feature_conf, model_conf, feature_map

def compute_user_item_feature_dims(feat_conf, model_conf):
    return None, feat_conf["item_embedding_dim"]

#精度对齐
def get_unique_path(file_path):
    """
    如果文件存在，则在文件名后添加数字后缀（如 _1, _2），返回新的 Path 对象。
    """
    path = Path(file_path)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1

def to_clean_cpu_tensor(data: Any) -> Any:
    """
    递归地将数据结构中的所有PyTorch张量移动到CPU并与计算图分离。
    支持处理张量、列表、元组和字典。
    """
    if isinstance(data, torch.Tensor):
        # 1. 确保张量在CPU上并与计算图分离
        return data.detach().cpu()
    elif isinstance(data, (list, tuple)):
        # 2. 如果是列表或元组，递归处理其每个元素
        return type(data)(to_clean_cpu_tensor(item) for item in data)
    elif isinstance(data, dict):
        # 3. 如果是字典，递归处理其每个值
        return {key: to_clean_cpu_tensor(value) for key, value in data.items()}
    else:
        # 4. 如果是其他类型，原样返回
        return data
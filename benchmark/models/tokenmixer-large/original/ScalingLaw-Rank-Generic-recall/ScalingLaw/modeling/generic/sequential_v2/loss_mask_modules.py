import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import json
from typing import Dict, List, Tuple
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.generic.sequential_v2.features import SequentialFeatures
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


@ModelRegistry.register()
class DefaultLossMask(BaseModel):
    
    """
    Construct loss mask based on past_payloads feature name
    """
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "default"

    def forward(self, model_inputs: Dict[str, torch.Tensor]):
        supervision_ids = model_inputs['past_ids']
        loss_weights = model_inputs["loss_weights"]
        return self.name, (supervision_ids != 0) * loss_weights

@ModelRegistry.register(req_hp=True)
class FeatureBasedLossMask(BaseModel):
    
    """
    Construct loss mask based on past_payloads feature name

    "hp": {
        "feature_name": str,
        "mode": "exclude" or "include",
        "target_value": int}
    """
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "feature_based_loss_mask"
        self.id_name = model_cfg[Const.HP].get("feature_name")
        self.mode = model_cfg[Const.HP].get("mode")
        target_values: List[str] = model_cfg[Const.HP].get("target_value")
        feture_map = self.load_feature_map(common_hp["export_conf"])
        self.target_ids = []
        for target_value in target_values:
            target_id = feture_map.get(target_value, None)
            if target_id is None:
                logging.error("target_value %s does not exist in the feature map, please check.", target_value)
            else:
                self.target_ids.append(int(target_id))
        
    def load_feature_map(self, export_conf):
        save_dir = export_conf["model_output_dir"]
        with open("%s/%s.config/feature_map.json" % (save_dir, export_conf["save_dir_name"]), 'r', encoding="utf-8") as fp:
            feture_map = json.load(fp)
        return feture_map

    def forward(self, model_inputs: Dict[str, torch.Tensor]):
        feature_id = model_inputs[self.id_name]
        list_ids = torch.tensor(self.target_ids, dtype=torch.int64).to(feature_id.device)
        if self.mode == "exclude":
            return self.name, (feature_id.unsqueeze(-1) != list_ids).any(-1)
        elif self.mode == "include":
            return self.name, (feature_id.unsqueeze(-1) == list_ids).any(-1)
        else:
            logging.error("mode should be either include or exclude.")


@ModelRegistry.register(req_hp=True)
class AGFeatureBasedLossMask(BaseModel):
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.name = "ag_feature_based_loss_mask"
        self.id_name = model_cfg[Const.HP].get("feature_name")
        self.id_name_2 = model_cfg[Const.HP].get("feature_name_2", "supervise_list")
        self.id_name_3 = model_cfg[Const.HP].get("feature_name_3", "list_id_list")
        self.mode = model_cfg[Const.HP].get("mode")
        self.target_value = model_cfg[Const.HP].get("target_value", 0)
        self.target_value_2 = model_cfg[Const.HP].get("target_value_2", 0)
        self.target_value_3 = model_cfg[Const.HP].get("target_value_3", 0)
        self.sample_ratio = model_cfg[Const.HP].get("sample_ratio", 1.0)

    def forward(self, model_inputs: Dict[str, torch.Tensor]):
        feature_id = model_inputs[self.id_name]
        list_ids = torch.tensor(self.target_value, dtype=torch.int64).to(feature_id.device)
        if self.mode == "exclude":
            return self.id_name, (feature_id.unsqueeze(-1) != list_ids).any(-1)
        elif self.mode == "include":
            if isinstance(self.target_value, list):
                # 如果 target_value 是一个列表，使用 torch.isin 判断 feature_id 是否在 target_value 中
                return self.id_name, torch.isin(feature_id, torch.tensor(self.target_value).to(feature_id.device))
            else:
                # 如果 target_value 只是一个值，按原逻辑处理
                return self.id_name, (feature_id.unsqueeze(-1) == list_ids).any(-1)
        elif self.mode == "sample":
            feature_id_2 = model_inputs[self.id_name_2] #feature_id_2 是 label
            condition = (feature_id == self.target_value) & (feature_id_2 == self.target_value_2) #like tensor([True, False, False])
            mask = torch.ones_like(feature_id, dtype=torch.bool).to(feature_id.device)
            # 如果有符合条件的索引，进行随机采样
            if condition.any(): #如果有任何True
                # 获取符合条件的二维索引
                select_indices = torch.nonzero(condition)  # 返回元组形式的索引
                #select_indices like tensor([[2], [4]])

                # 计算采样数量
                sample_num = max(1, round(len(select_indices[0]) * self.sample_ratio))  # 至少采样一个

                # 随机选择索引
                sampled_indices = select_indices[torch.randperm(len(select_indices))[:sample_num]]  # 确保唯一性

                # 更新掩码
                mask[condition] = False  # 将所有符合条件的索引设为 False
                mask[sampled_indices[:, 0], sampled_indices[:, 1]] = True  # 将采样出的索引设为 True
                    # 如果 mode 是 "sample"，表示数据已被采样，因此直接返回采样后的结果
            return self.id_name, mask
        elif self.mode == "sample_new":
            feature_id_2 = model_inputs[self.id_name_2]
            feature_id_3 = model_inputs[self.id_name_3]
            condition = (feature_id == self.target_value) & (feature_id_2 == self.target_value_2) & torch.isin(feature_id_3, torch.tensor(self.target_value_3).to(feature_id.device))# (64,400)
            mask = torch.ones_like(feature_id, dtype=torch.bool).to(feature_id.device)
            # 如果有符合条件的索引，进行随机采样
            if condition.any():
                # 获取符合条件的二维索引
                select_indices = torch.nonzero(condition)  # 返回元组形式的索引, 在condition矩阵中的坐标 shape：（545，2）

                # 计算采样数量
                sample_num = max(1, round(len(select_indices) * self.sample_ratio))  # 至少采样一个 

                # 随机选择索引
                sampled_indices = select_indices[torch.randperm(len(select_indices))[:sample_num]]  # 确保唯一性

                # 更新掩码
                mask[condition] = False  # 将所有符合条件的索引设为 False
                mask[sampled_indices[:, 0], sampled_indices[:, 1]] = True  # 将采样出的索引设为 True
                    # 如果 mode 是 "sample"，表示数据已被采样，因此直接返回采样后的结果
            return self.id_name, mask
        else:
            logging.error("loss_mask_conf.mode should be either include or exclude.")
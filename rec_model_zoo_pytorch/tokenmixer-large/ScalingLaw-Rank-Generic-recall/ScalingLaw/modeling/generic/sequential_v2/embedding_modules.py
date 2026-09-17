import abc
import logging
from typing import Dict, List, Tuple
 
import torch
import os
import pandas as pd
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.generic.initialization import truncated_normal
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const

 
 
class EmbeddingModule(BaseModel):
 
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
 
    @abc.abstractmethod
    def get_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass
 
    @abc.abstractmethod
    def get_candidate_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass
 
    @abc.abstractmethod
    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass
 
    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass
 
@ModelRegistry.register()
class LocalEmbeddingModuleWithSideInfo(EmbeddingModule):
    """
    带有sideinfo的Embedding模块, 用于生成物品和用户的表示。
    """
 
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp["feature_conf"]
        item_feature_columns: Dict = feat_conf.get('item_feature_columns', None)

        user_feature_columns: Dict = feat_conf.get('user_feature_columns', None)
        train_conf = common_hp["train_conf"]
        self._item_id_name = feat_conf.get("infer_items_key", "sequence_item_ids")
        self.use_trained_emb = train_conf.get("use_trained_emb", False)
        self.is_recall = train_conf.get('is_recall', False)
        self.multi_user = train_conf.get('multi_user', False)
        self.use_item_emb_mlp = train_conf.get('use_item_emb_mlp', True)
        if self.multi_user == True:
            user_multi_emb_mlp = {}
        self.recall_feat_dict = train_conf.get("save_emb_dict", {})

 
        self.padding_index = feat_conf.get("padding_index", 0)
        self.multi_value_prefix = feat_conf.get('multi_value_prefix', "pref_")
        if user_feature_columns is None or item_feature_columns is None:
            raise ValueError("user_feature_columns and item_feature_columns cannot be None")
        # 物品信息嵌入字典
        item_info_embs = {}
        candidate_item_info_embs = {}
        self._item_feature_names: List[str] = []
        self._multivalue_item_feature_names: List[str] = []
        self._candidate_item_feature_names: List[str] = []
        self.item_dyte: Dict[str, str] = {}
        # 用户信息嵌入字典
        user_info_embs = {}
        self._user_feature_names: List[str] = []
        self.user_dyte: Dict[str, str] = {}

        # 初始化物品特征嵌入
        self.enabled_item_features: List[str] = []
        for feature_name, feature_info in item_feature_columns.items():
            feature_count = item_feature_columns[feature_name].get('feature_count', 10)
            feature_dim = item_feature_columns[feature_name].get('dim', 32)
            feature_enabled = item_feature_columns[feature_name].get("enabled", True)
            feature_dtype = item_feature_columns[feature_name].get("dtype", "int")
            is_multivalue = item_feature_columns[feature_name].get("is_multivalue", False)
            self.item_dyte[feature_name] = feature_dtype
            if feature_enabled == True:
                self.enabled_item_features.append(feature_name)
                # 在精排任务中如果启用 use_trained_emb，则给item id使用先前召回任务的参数 加入item id外的特征 增幅
                if feature_name in self.recall_feat_dict.keys()  and self.use_trained_emb == True and self.is_recall == False:
                    save_emb_dir = train_conf.get("save_emb_dir", None)
                    _item_emb_table = torch.load(save_emb_dir)[feature_name]
                    for param in _item_emb_table.parameters():
                        with torch.no_grad():
                            param_numpy = param.detach().cpu().numpy()
                            param_cpu = torch.from_numpy(param_numpy)
                    _item_emb_table_cpu = torch.nn.Embedding(feature_count + 1, self.recall_feat_dict[feature_name], padding_idx=self.padding_index)
                    for param in _item_emb_table_cpu.parameters():
                        with torch.no_grad():
                            param[:] = param_cpu
                        # 是否关闭参数更新
                        if True:
                           param.requires_grad = False
                           logging.info(f"close the param grad of {feature_name}")
                    item_info_embs[feature_name] = _item_emb_table_cpu
                    candidate_item_info_embs['candidate_' + feature_name] = _item_emb_table_cpu
                    logging.info(f"Use the recall {feature_name} embedding")
                elif feature_dtype == "con":
                    # 处理商品特征里的离散特征
                    _con_layer = torch.nn.BatchNorm1d(1)
                    item_info_embs[feature_name] = _con_layer
                    candidate_item_info_embs['candidate_' + feature_name] = _con_layer
                elif feature_dtype == "int":
                    _item_emb_table = torch.nn.Embedding(feature_count + 1, feature_dim, padding_idx=self.padding_index)
                    item_info_embs[feature_name] = _item_emb_table
                    candidate_item_info_embs['candidate_' + feature_name] = _item_emb_table
                else:
                    logging.error("feature_dtype %s is undefined for %s.", feature_dtype, feature_name)
                if is_multivalue:
                    self._multivalue_item_feature_names.append(feature_name)
                self._item_feature_names.append(feature_name)
                self._candidate_item_feature_names.append('candidate_' + feature_name)
        
        if self.use_trained_emb == True and self.is_recall == False:
            for feature_name in self.recall_feat_dict.keys():
                _item_emb_table = torch.nn.Embedding(item_feature_columns[feature_name]["feature_count"] + 1, item_feature_columns[feature_name]["dim"], padding_idx=self.padding_index)
                item_info_embs["extra_" + feature_name] = _item_emb_table
                candidate_item_info_embs['extra_candidate_' + feature_name] = _item_emb_table
                self._item_feature_names.append("extra_" + feature_name)
                self._candidate_item_feature_names.append('extra_candidate_' + feature_name)
 
        # 初始化用户特征嵌入
        self.enabled_user_features: List[str] = []
        for feature_name, feature_info in user_feature_columns.items():
            feature_count = user_feature_columns[feature_name].get('feature_count', 10)
            feature_dim = user_feature_columns[feature_name].get('dim', 32)
            feature_enabled = user_feature_columns[feature_name].get("enabled", True)
            feature_dtype = user_feature_columns[feature_name].get("dtype", "int")
            associated_item_feature = user_feature_columns[feature_name].get("associated", None)
            self.user_dyte[feature_name] = feature_dtype
            if feature_enabled == True:
                self.enabled_user_features.append(feature_name)
                if feature_dtype == "pref":
                    # 默认的关联特征是去掉pref_前缀的商品特征名，例如pref_artist默认的关联特征是artist
                    associated = feature_name.replace(self.multi_value_prefix,
                                                      "") if associated_item_feature is None \
                        else associated_item_feature
                    if associated in self.enabled_item_features:
                        # 若关联特征是item embedding 且启用
                        user_info_embs[feature_name] = item_info_embs.get(associated)
                    else:
                        logging.error(
                            "The assoicated feature %s for %s is not an item feature or not enabled in config.",
                            associated, feature_name)
                elif feature_dtype == "int":
                    # 若无关联特征，直接初始化
                    user_info_embs[feature_name] = torch.nn.Embedding(feature_count + 1, feature_dim,
                                                                      padding_idx=self.padding_index)
                else:
                    logging.error("feature_dtype %s is undefined for the user.", feature_dtype)
                self._user_feature_names.append(feature_name)
                if self.multi_user == True:
                    self._item_embedding_dim = common_hp["model_conf"].get("item_embedding_dim", 64)
                    if feature_dtype == "int":
                        user_multi_emb_mlp[feature_name] = torch.nn.Linear(feature_dim, self._item_embedding_dim)
                    elif feature_dtype == "pref":
                        associated = feature_name.replace(self.multi_value_prefix,
                                                      "") if associated_item_feature is None else associated_item_feature
                        if associated in self.enabled_item_features:
                            in_dim = item_feature_columns[associated]["dim"]
                            user_multi_emb_mlp[feature_name] = torch.nn.Linear(in_dim, self._item_embedding_dim)
        
        self._item_info_embs = torch.nn.ModuleDict(item_info_embs)
        self._user_info_embs = torch.nn.ModuleDict(user_info_embs)
        self._candidate_item_info_embs = torch.nn.ModuleDict(candidate_item_info_embs)
        if self.multi_user == True:
            self._user_multi_emb_mlp = torch.nn.ModuleDict(user_multi_emb_mlp)
 
        self._item_embedding_dim = common_hp["model_conf"].get("item_embedding_dim", 64)
 
        # 计算物品和用户输入维度
        item_input_dim = 0
        for k, v in item_info_embs.items():
            try:
                if item_feature_columns[k]["dtype"] == "con":
                    item_input_dim += 1
                else:
                    item_input_dim += v.weight.shape[1]
            except:
                if "extra" in k:
                    feature_name = k.removeprefix("extra_")
                    item_input_dim += item_feature_columns[feature_name]["dim"]
        user_input_dim = sum([v.weight.shape[1] for _, v in user_info_embs.items()])
 
 
        # 如果设置了输出维度，则创建MLP层
        self.item_emb_mlp, self.user_emb_mlp = torch.nn.Identity(), torch.nn.Identity()
 
        if self._item_embedding_dim:
            if self.use_item_emb_mlp:
                self.item_emb_mlp = torch.nn.Linear(item_input_dim, self._item_embedding_dim)
                logging.info('Set item_emb_mlp to Linear: %s -> %s' % (item_input_dim, self._item_embedding_dim))
            else:
                self.item_emb_mlp = None
                logging.info('item emb mlp omitted')
            if self.multi_user or len(self._user_feature_names) == 0:
                logging.info('Omitting user emb emlp')
                self.user_emb_mlp = None
            else:
                self.user_emb_mlp = torch.nn.Linear(user_input_dim, self._item_embedding_dim)
                logging.info('Set user_emb_mlp to Linear: %s -> %s' % (user_input_dim, self._item_embedding_dim))
        else:
            if item_input_dim != user_input_dim:
                raise RuntimeError('item_input_dim and user_input_dim mismatch! user_dim : %s, item_dim : %s' %
                                   (user_input_dim, item_input_dim))
            self._item_embedding_dim = item_input_dim
 
        self.reset_params()
 
    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
 
    def debug_str(self) -> str:
        return f"LocalEmbeddingModuleWithSideInfo"
 
    def reset_params(self):
        for name, params in self.named_parameters():
            not_init_names = [f"_item_info_embs.{text}.weight" for text in self.recall_feat_dict.keys()]
            if ('emb' in name and name not in not_init_names) or (name in not_init_names and (self.use_trained_emb == False or self.is_recall == True)):
                logging.info("Initialize %s as truncated normal: %s params" % (name, params.data.size()))
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                logging.info("Skipping initializing params %s - not configured" % name)
 
    def get_candidate_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据物品特征获取物品嵌入。
 
        :param item_features: 物品特征字典。
        :return: 物品嵌入张量。
        """
 
        feature_emb_list = []
        for feature_name in self._candidate_item_feature_names:
            original_feature_name = feature_name.replace("candidate_", "")
            if original_feature_name in self.enabled_item_features:
                item_feature_id = item_features[feature_name]
                if self.item_dyte.get(original_feature_name) == "con":
                    feature_value = self._candidate_item_info_embs[feature_name](
                        item_feature_id.unsqueeze(1)).transpose(1, 2)
                else:
                    feature_value = self._candidate_item_info_embs[feature_name](item_feature_id)
                if original_feature_name in self._multivalue_item_feature_names:
                    feature_value = torch.sum(feature_value, dim=-1)
                feature_emb_list.append(feature_value)
        
        if self.use_trained_emb == True and self.is_recall == False:
            for feature_name in self.recall_feat_dict.keys():
                item_feature_id = item_features["candidate_" + feature_name]
                feature_value = self._candidate_item_info_embs["extra_candidate_" + feature_name](item_feature_id)
                feature_emb_list.append(feature_value)
        
        feature_embs = torch.cat(feature_emb_list, dim=-1)
 
        if self._item_embedding_dim != 0 and self.use_item_emb_mlp:
            feature_embs = self.item_emb_mlp(feature_embs)
 
        return feature_embs
    
    def get_item_embeddings(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据物品特征获取物品嵌入。
        
        :param item_features: 物品特征字典。
        :return: 物品嵌入张量。
        """

        feature_emb_list = []
        for feature_name in self._item_feature_names:
            if feature_name in self.enabled_item_features:
                item_feature_id = item_features[feature_name]
                max_item_feature_id = torch.max(item_feature_id)
                if self.item_dyte[feature_name] == "con":
                    feature_value = self._item_info_embs[feature_name](item_feature_id.unsqueeze(1)).transpose(1, 2)
                else:
                    feature_value = self._item_info_embs[feature_name](item_feature_id)
                if feature_name in self._multivalue_item_feature_names:
                    feature_value = torch.sum(feature_value, dim=-1)
                feature_emb_list.append(feature_value)
        
        if self.use_trained_emb == True and self.is_recall == False:
            for feature_name in self.recall_feat_dict.keys():
                item_feature_id = item_features[feature_name]
                feature_value = self._item_info_embs["extra_" + feature_name](item_feature_id)
                feature_emb_list.append(feature_value)
                
        feature_embs = torch.cat(feature_emb_list, dim=-1)
        
        if self._item_embedding_dim != 0 and self.item_emb_mlp is not None:
            feature_embs = self.item_emb_mlp(feature_embs)
 
        return feature_embs
 
    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据用户特征获取用户嵌入。
        
        :param user_features: 用户特征字典。
        :return: 用户嵌入张量。
        """
        eps = 1e-6
        feature_emb_list = []
        
        if len(self._user_feature_names) == 0:
            bs = user_features[self._item_id_name].shape[0]
            ret = torch.zeros((bs,self._item_embedding_dim), device=user_features[self._item_id_name].device)
            return ret

        for feature_name in self._user_feature_names:
            if feature_name in self.enabled_user_features:
                user_feature_id = user_features[feature_name]
                max_item_feature_id = torch.max(user_feature_id)
                if self.user_dyte[feature_name] == "pref":
                    # (B, M, emb_size) M 是多值特征padding后的长度
                    feature_values = self._user_info_embs[feature_name](user_feature_id)
                    feature_dim = feature_values.size(-1)
                    feat_mask = user_feature_id != self.padding_index
                    feat_mask = feat_mask.unsqueeze(-1).repeat(1, 1, feature_dim)
                    # 对pref多值特征做mean pooling，并忽略掉值为self.padding_index的填充index
                    feature_value = (feature_values * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + eps)
                else:
                    feature_value = self._user_info_embs[feature_name](user_feature_id)

                feature_emb_list.append(feature_value)

        feature_embs = torch.cat(feature_emb_list, dim=-1)

        if self._item_embedding_dim != 0:
            feature_embs = self.user_emb_mlp(feature_embs)

        return feature_embs
    
    def get_item_embeddings_only(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据物品特征获取物品嵌入。
        
        :param item_features: 物品特征字典。
        :return: 物品嵌入张量。
        """
 
        item_feature_id = item_features[self._item_id_name]
        return self._item_info_embs[self._item_id_name](item_feature_id)
 
    # 获取全部item的embedding
    def get_all_item_id_only_embeddings(self, item_id) -> torch.Tensor:
        if isinstance(item_id, torch.Tensor):
            pass
        elif isinstance(item_id, dict):
            item_id = item_id[self._item_id_name]
        else:
            raise ValueError("item id must be dict or Tensor")
        return self._item_info_embs[self._item_id_name](item_id)
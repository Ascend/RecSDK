import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, List, Tuple
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
import logging

@ModelRegistry.register()
class FeedForwardModule(BaseModel):
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        
        if not Const.SUB_MODELS in model_cfg:
            logging.error("You must assign at least one sub module for FeedForwardModule in the config.")

        self.prediction_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()

        ])
    
    def forward(self, x):
        predictions = dict()
        for ffn in self.prediction_modules:
            pred_name, pred = ffn(x)
            predictions[pred_name] = pred
        return predictions

@ModelRegistry.register()     
class FeedForwardModuleForRerankScore(BaseModel):
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self.name = "rerank_score"
        input_dim = model_conf.get("item_embedding_dim", 128)
        
        if model_conf.get('use_user_embeddings_for_rerank', False):
            input_dim *= 2

        self.feed_forward = torch.nn.Linear(in_features=input_dim, out_features=input_dim)
        self.out_layer = torch.nn.Linear(in_features=input_dim,out_features=1)
        self.layer_norm_1 = torch.nn.LayerNorm([input_dim], eps=1e-7)
        self.layer_norm_2 = torch.nn.LayerNorm([input_dim], eps=1e-7)
        self.act = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        x = x + self.act(self.feed_forward(self.layer_norm_1(x)))
        x = self.out_layer(self.layer_norm_2(x))
        x = torch.sigmoid(x)
        return self.name, x

@ModelRegistry.register()      
class FeedForwardModuleForNextActionPred(BaseModel):
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self.name = "next_action_prob"
        input_dim = model_conf.get("item_embedding_dim", 128)
        output_dim = model_conf.get("num_ratings", 5)
        
        if model_conf.get('use_user_embeddings_for_rerank', False):
            input_dim *= 2
        
        self.feed_forward = torch.nn.Linear(in_features=input_dim, out_features=input_dim)
        self.out_layer = torch.nn.Linear(in_features=input_dim,out_features=output_dim + 1)
        self.layer_norm_1 = torch.nn.LayerNorm([input_dim], eps=1e-7)
        self.layer_norm_2 = torch.nn.LayerNorm([input_dim], eps=1e-7)
        self.act = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        x = x + self.act(self.feed_forward(self.layer_norm_1(x)))
        x = self.out_layer(self.layer_norm_2(x))
        x = torch.sigmoid(x)
        return self.name, x
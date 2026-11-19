from typing import Dict, Mapping, Any
from torch import nn
import sys
import logging
 
from modeling.generic.utils.constants import Const
 
class BaseModel(nn.Module):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        所有模块化类的初始化所需的参数，都包含在model_cfg、common_hp、model_cls_dict三个参数
        所有子类都必须实现该初始化方法，这是动态加载的核心
        :param model_cfg: 模型配置类，通过配置文件获取
        :param common_hp: 通用超参，通过配置文件获取
        :param model_cls_dict: 模型class对象字典，key为模块名，value为模块类实例
        """
        super(BaseModel, self).__init__()
        self.model_cfg = model_cfg
        self.common_hp = common_hp
        self.model_cls_dict = model_cls_dict
 
    @staticmethod
    def init_model(model_cfg: Dict, common_hp: Dict, model_cls_dict: Any):
        """
        初始化model_cfg对应模块模型类
        :param model_cfg: 模型配置类，通过配置文件获取
        :param common_hp: 通用超参，通过配置文件获取
        :param model_cls_dict: 模型class对象字典
        :return: 模型类对象
        """
        model_cls = model_cls_dict[model_cfg[Const.MODULE_NAME]]
        model = model_cls(model_cfg=model_cfg,
                          common_hp=common_hp,
                          model_cls_dict=model_cls_dict)
        if hasattr(model, '_load_state_dict_post_hook'):
            model.register_load_state_dict_post_hook(model._load_state_dict_post_hook(common_hp))
        return model
 
    def init_sub_model(self, sub_key: str):
        """
        初始化子模块模型类
        :param model_cfg: 模型配置类，通过配置文件获取
        :param common_hp: 通用超参，通过配置文件获取
        :param model_cls_dict: 模型class对象字典，key为模块名，value为模块类实例
        :param name: 初始化子模块名
        :return: 子模块模型类对象
        """
        if sub_key not in self.model_cfg.get(Const.SUB_MODELS, {}):
            return None
        sub_model_cfg = self.model_cfg[Const.SUB_MODELS][sub_key]
        return BaseModel.init_model(model_cfg=sub_model_cfg, common_hp=self.common_hp, model_cls_dict=self.model_cls_dict)
 
 
    def _load_state_dict_post_hook(self, common_hp: Dict):
        """
        生成加载权重参数后置钩子函数
        :param common_hp: 通用超参，通过配置文件获取，可手动添加后置初始化所需参数
        :return: 加载权重参数后置钩子函数
        """
 
        def hook(state_dict, prefix):
            """
            加载权重参数后置钩子函数
            :param state_dict: 模型类对象
            :param prefix: 缺失和不在预期内的权重参数
            """
            pass
 
        return hook
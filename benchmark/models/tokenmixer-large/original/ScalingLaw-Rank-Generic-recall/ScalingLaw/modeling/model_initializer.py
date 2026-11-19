from typing import Dict, Callable, Any
import json
import os
import importlib.util
import copy
import torch
import logging
 
from modeling.model_registry import ModelRegistry
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.generic.utils.constants import Const
 
REF_ALGO_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "ScalingLaw"))
 
 
class ModelInitializer:
 
    @staticmethod
    def init(gr_module_cfg: dict):
        module_cfg = gr_module_cfg[Const.MODEL_CFG]
        common_hp = gr_module_cfg[Const.COMMON_HP]
 
        module_cfg_checker = ModuleCfgChecker()
        module_cfg_checker.check(module_cfg=module_cfg,
                                 root_path=REF_ALGO_ROOT_PATH)
        return BaseModel.init_model(model_cfg=module_cfg,
                                       common_hp=common_hp,
                                       model_cls_dict=ModelRegistry.get_module_cls_dict())
 
 
class ModuleCfgChecker:
    def __init__(self):
        self.errors = []  # 存储所有校验错误
 
    def check_and_record(self, is_valid: bool, error_log: str, *args):
        if is_valid:
            return
        self.errors.append(error_log.format(*args))
 
    def check(self, module_cfg: Dict, root_path: str):
        self.__check_module_cfgs(module_cfg, root_path)
        if self.errors:
            logging.error("Model configuration may have errors: %s", self.errors)
 
    def __check_module_cfgs(self, module_cfg: Dict, root_path: str):
        if not module_cfg:
            return
        self.__check_module_cfg(module_cfg=module_cfg,
                                module_meta_dict=ModelRegistry.get_module_meta_dict(),
                                root_path=root_path)
        biz_sub_cfg = module_cfg.get(Const.SUB_MODELS, {})
        for key, value in biz_sub_cfg.items():
            self.__check_module_cfgs(module_cfg=biz_sub_cfg[key], root_path=root_path)
 
    def __check_module_cfg(self, module_meta_dict: Dict, module_cfg: Dict, root_path):
        is_customize = module_cfg.get(Const.IS_CUSTOMIZE, False)
 
        # 校验必须字段是否存在
        required_keys = {Const.MODULE_NAME, Const.PATH, Const.CLS_NAME, Const.IS_CUSTOMIZE} if is_customize else {Const.MODULE_NAME}
        self.check_and_record(required_keys.issubset(set(module_cfg.keys())),"'{}' miss required fields, required fields: {}, configured fields: {}.", required_keys, module_cfg.keys())
 
        if is_customize:
            try:
                ModelRegistry.register_outer_module_cls(root_path=root_path,
                                                        py_path=module_cfg[Const.PATH],
                                                        cls_name=module_cfg[Const.CLS_NAME],
                                                        module_name=module_cfg[Const.MODULE_NAME])
            except Exception as e:
                self.errors.append(f"{e}")
            finally:
                return
 
        module_name = module_cfg[Const.MODULE_NAME]
        self.check_and_record(module_name in module_meta_dict, "'{}' is not customized and does not exist in current algorithm packet.", module_name)
        module_meta = module_meta_dict[module_name]
        self.check_and_record(not module_meta.req_hp or module_cfg.get(Const.HP, {}), "'{}' miss 'hp' field", module_name)
 
        cfg_subs = set(module_cfg.get(Const.SUB_MODELS, {}).keys())
        cfg_subs = self.__check_req_subs(module_name=module_name, req_subs=module_meta.req_subs, cfg_subs=cfg_subs)
        cfg_subs = self.__check_multi_sel_one_subs(cfg_subs=cfg_subs, module_name=module_name,
                                                   multi_sel_one_subs=module_meta.multi_sel_one_subs)
        cfg_subs = self.__check_multi_sel_multi_subs(module_name=module_name, cfg_subs=cfg_subs,
                                                     multi_sel_multi_subs=module_meta.multi_sel_multi_subs)
        self.__check_opt_subs(cfg_subs=cfg_subs, module_name=module_name, opt_subs=module_meta.opt_subs)
 
    def __check_opt_subs(self, cfg_subs, module_name, opt_subs):
        if not opt_subs:
            self.check_and_record(not cfg_subs, "'{}' has non define subs: {}", module_name, cfg_subs)
            return
 
        non_define_subs = cfg_subs.difference(opt_subs)
        self.check_and_record(not non_define_subs, "'{}' has non define subs: {}", module_name, non_define_subs)
 
    def __check_multi_sel_multi_subs(self, module_name, cfg_subs, multi_sel_multi_subs):
        for multi_sel_multi_sub in multi_sel_multi_subs:
            intersection_subs = multi_sel_multi_sub & cfg_subs
            self.check_and_record(len(intersection_subs) > 0, "'{}' miss multi_sel_multi_subs: {}",
                                  module_name, multi_sel_multi_sub)
            cfg_subs = cfg_subs.difference(intersection_subs)
        return cfg_subs
 
    def __check_req_subs(self, module_name: str, req_subs: set, cfg_subs: set):
        self.check_and_record(req_subs.issubset(cfg_subs), "'{}' miss req subs: {}", module_name, req_subs.difference(cfg_subs))
        return cfg_subs.difference(req_subs)
 
    def __check_multi_sel_one_subs(self, cfg_subs, module_name, multi_sel_one_subs):
        for multi_sel_one_sub in multi_sel_one_subs:
            intersection_subs = multi_sel_one_sub & cfg_subs
            self.check_and_record(len(intersection_subs) != 1,
                                  "'{}' has less or more multi_sel_one_subs: '{}'.", module_name,
                                  intersection_subs)
            cfg_subs = cfg_subs.difference(intersection_subs)
        return cfg_subs
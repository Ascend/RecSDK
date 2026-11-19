import importlib
from typing import Dict, Callable, Any
from torch import nn
import sys
from dataclasses import dataclass, field
from typing import Any, Dict
from enum import Enum
import os
import importlib.util
import logging
import json
from modeling.generic.utils.constants import Const
import inspect
 
 
@dataclass
class GrModuleMeta:
    name: str  # 模块名
    req_hp: bool  # 是否必须配置超参
    opt_subs: set  # 可选子模块
    req_subs: set  # 必选子模块
    multi_sel_multi_subs: list[set]  # 多选子模块
    multi_sel_one_subs: list[set]  # 多选一子模块
    module_cls: Any  # 模型class实例
 
 
class ModelRegistry:
    __module_meta_dict: Dict[str, GrModuleMeta] = {}
    __outer_module_cls_dict: Dict[str, Any] = {}
 
    @staticmethod
    def register(name=None,
                 req_subs: set = None,
                 opt_subs: set = None,
                 multi_sel_multi_subs: list[set] = None,
                 multi_sel_one_subs: list[set] = None,
                 req_hp=False):
        """
        注册模块化模型类，通过注册信息校验配置文件格式是否正确
 
        :param name: 模块名
        :param req_subs: 是否必须配置超参
        :param opt_subs: 可选子模块，这个字段定义的子模块配不配置都可以。格式: {"sub1", "sub2", ...}
        :param req_hp: 必选子模块，这个字段定义的模块必须配置。格式: {"sub1", "sub2", ...}
        :param multi_sel_multi_subs: 多选子模块，这个字段定义的模块必须且至少配置一个，可多配置，支持同时配置多组多选子模块。格式: [{"sub1-1", "sub1-2"}, {"sub2-1", "sub2-2", "sub1-3"}]
        :param multi_sel_one_subs: 多选一子模块，这个字段定义的模块必选且只能配置一个，支持同时配置多组多选一子模块。格式: [{"sub1-1", "sub1-2"}, {"sub2-1", "sub2-2", "sub1-3"}]
        :return: 模块注册装饰器函数
        """
        opt_subs = opt_subs if opt_subs else set()
        req_subs = req_subs if req_subs else set()
        multi_sel_multi_subs = multi_sel_multi_subs if multi_sel_multi_subs else list()
        multi_sel_one_subs = multi_sel_one_subs if multi_sel_one_subs else list()
 
        def decorator(cls):
            module_name = name if name else cls.__name__
            if module_name in ModelRegistry.__module_meta_dict:
                logging.info("'%s' already register.", module_name)
 
            gr_module_meta = GrModuleMeta(name=module_name,
                                          req_subs=req_subs,
                                          opt_subs=opt_subs,
                                          multi_sel_multi_subs=multi_sel_multi_subs,
                                          multi_sel_one_subs=multi_sel_one_subs,
                                          module_cls=cls,
                                          req_hp=req_hp)
            ModelRegistry.__module_meta_dict[module_name] = gr_module_meta
            logging.info("'%s' register succeed.", module_name)
            return cls
 
        return decorator
    
    @staticmethod
    def register_all_modules(modul_dir):
        """
        递归加载指定目录及其子目录下的所有 Python 文件并返回模块列表。
 
        参数：
            directory (str): 模块目录的路径。
 
        返回：
            list: 包含所有加载的模块的列表。
        """
        for root, dirs, files in os.walk(modul_dir):
            # 遍历当前目录下的所有文件
            for filename in files:
                if filename == "torchrec_trainer.py":
                    continue
                # 检查是否是 Python 文件
                if filename.endswith('.py'):
                    # 构造文件路径
                    file_path = os.path.join(root, filename)
                    # 获取模块名（去掉 .py 扩展名）
                    module_name = os.path.splitext(filename)[0]
                    # 创建模块规格
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None:
                        continue
                    # 创建模块
                    module = importlib.util.module_from_spec(spec)
                    try:
                        # 执行模块
                        spec.loader.exec_module(module)
                    except Exception as e:
                        logging.error("Loading module: '%s' error, %s.", module_name, e)
                        raise e
 
    @staticmethod
    def get_module_cls_dict():
        module_cls_dict = {}
        for name, gr_module_meta in ModelRegistry.__module_meta_dict.items():
            module_cls_dict[name] = gr_module_meta.module_cls
 
        conv_module_names = set(module_cls_dict.keys())
        outer_module_names = set(ModelRegistry.__outer_module_cls_dict.keys())
 
        conflict_module_names = conv_module_names & outer_module_names
        if conflict_module_names:
            logging.warning("Customize modules conflicts with algorithm modules: %s.", conflict_module_names)
        module_cls_dict.update(ModelRegistry.__outer_module_cls_dict)
        return module_cls_dict
 
    @staticmethod
    def get_module_meta_dict():
        return ModelRegistry.__module_meta_dict
 
    @staticmethod
    def register_outer_module_cls(root_path, py_path, cls_name, module_name):
        outer_module_cls = ModelRegistry.__get_module_from_outer(root_path, py_path, cls_name)
        ModelRegistry.__outer_module_cls_dict[module_name] = outer_module_cls

    @staticmethod
    def __get_module_from_outer(root_path, py_path, class_name):
        """
                    从指定的项目根目录中加载指定的 Python 文件，并获取其中的类对象。
 
                    参数：
                        root_dir (str): 项目根目录的路径。
                        file_path (str): 相对于项目根目录的 Python 文件路径。
                        class_name (str): 要获取的类的名称。
 
                    返回：
                        class_obj: 指定的类对象。
 
                    异常：
                        FileNotFoundError: 如果指定的文件或目录不存在。
                        ImportError: 如果无法导入文件或其中的类。
                        AttributeError: 如果在文件中找不到指定的类。
                    """
        # 检查项目根目录是否存在
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"The root directory:'{root_path}' does not exist.")
 
        # 检查目标文件是否存在
        abs_file_path = os.path.join(root_path, py_path)
        if not os.path.exists(abs_file_path):
            raise FileNotFoundError(f"The target file:'{py_path}' does not exist.")
 
        try:
            # 获取模块名（从文件路径中提取）
            module_name = os.path.splitext(os.path.basename(py_path))[0]
 
            # 创建模块规格
            spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
 
            if spec is None:
                raise ImportError(f"Unable to load file: '{abs_file_path}'")
 
            # 创建模块
            module = importlib.util.module_from_spec(spec)
 
            try:
                # 执行模块
                spec.loader.exec_module(module)
            except Exception as e:
                raise ImportError(f"Error loading file:'{abs_file_path}', exception: {e}")
 
            # 获取类对象
            class_obj = getattr(module, class_name, None)
            # print("module: ", module, "class_name: ", class_name)
 
            if class_obj is None:
                raise AttributeError(f"Unable to find class:'{class_name}' in file:'{py_path}'.")
            return class_obj
        finally:
            pass
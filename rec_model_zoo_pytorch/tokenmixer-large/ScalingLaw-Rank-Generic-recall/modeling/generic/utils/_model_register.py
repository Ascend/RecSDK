from typing import Dict, Any
import logging
MODEL_CLS_DICT: Dict[str, Any] = {}

def register_model_cls(name=None):
    def decorator(cls):
        # 允许自定义名称，否则用类名
        model_name = name if name else cls.__name__
        if model_name in MODEL_CLS_DICT:
            logging.info(f"模型 {model_name} 已经存在！")
        MODEL_CLS_DICT[model_name] = cls
        logging.info(f"注册模型： {model_name}")
        return cls
    return decorator
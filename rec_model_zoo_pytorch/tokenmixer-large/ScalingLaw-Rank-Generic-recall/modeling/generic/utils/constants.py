from typing import Tuple

import torch

L2NORM_STRING = "l2"
LAYERNORM_STRING = "ln"
MAX_K = 2500

class Const:
    """
    定义配置文件以及框架中使用的常量
    """
    BEST_LOSS = 1e+7
    EPS = 1e-6
    TransformerCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    LABEL_DICT = []
    MODULE_NAME = "name"
    IS_CUSTOMIZE = "is_customize"
    PATH = "path"
    CLS_NAME = "cls_name"
    HP = "hp"
    SUB_MODELS = "sub_models"
    IS_POST_INIT = "is_post_init"
    MODEL_CFG = "model_cfg"
    COMMON_HP = "common_hp"
    EXPECTED_NUM_UNIQUE_ITEMS = 499999
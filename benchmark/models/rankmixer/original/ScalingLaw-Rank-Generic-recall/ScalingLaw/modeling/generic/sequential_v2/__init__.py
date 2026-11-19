import os

import torch
from utils.common_utils import logging


HAS_FBGEMM_OPS = False
try:
    import fbgemm_gpu
    torch.ops.load_library(os.path.join(os.path.dirname(fbgemm_gpu.__file__), "fbgemm_gpu_py.so"))
    HAS_FBGEMM_OPS = True
    logging.info("MXRec fbgemm ops installed, ready to be called.")
except Exception as e:
    logging.info("MXRec fbgemm ops not installed: %s",  e)


HAS_TORCHREC = False
try:
    import torchrec
    HAS_TORCHREC = HAS_FBGEMM_OPS
    if HAS_TORCHREC:
        logging.info("torchrec and required ops installed, set 'model_cfg.type'='GRModelEp' to enable.")
    else:
        logging.info("Required ops of torchrec not installed.")
except Exception as e:
    logging.info("torchrec not installed: %s",  e)


HAS_ATTN_FUSION_OPS = False
try:
    torch.ops.load_library(os.path.join(os.path.dirname(torch.__file__), "libhstu_dense_ops.so"))
    HAS_ATTN_FUSION_OPS = True
    logging.info("MXRec attention jagged fusion ops installed, enabled by default.")
except Exception as e:
    logging.info("MXRec attention jagged fusion ops not installed: %s",  e)
# coding: UTF-8
from .util.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION
from .util.tf_version_adapter import npu_ops, hccl_ops
from .saver.patch import patch_for_saver
from .graph.patch import patch_for_dataset


patch_for_saver()
patch_for_dataset()

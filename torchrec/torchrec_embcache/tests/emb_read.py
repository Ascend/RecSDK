#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import numpy as np

from acc_test.util import setup_logging


setup_logging(0)


def emb_read(out_dir="save_dir", table_names=None, ranks=None):
    if table_names is None:
        table_names = ["table0", "table1"]
    if ranks is None:
        ranks = [0]

    def do_in(table_name, rank, slice_name, dtype_in=np.float32):
        attribute_path = os.path.join(out_dir, table_name, f"rank{rank}", slice_name, "slice.attribute")
        data_path = os.path.join(out_dir, table_name, f"rank{rank}", slice_name, "slice.data")
        attr = np.fromfile(attribute_path, dtype=np.int64)
        data = np.fromfile(data_path, dtype=dtype_in).reshape(*attr[1:])
        return attr, data

    embedding_dt = dict()
    for table in table_names:
        tabledt = dict()
        logging.info("Processing table: %s", table)
        for rank in ranks:
            rankdt = dict()
            logging.debug("--------------------rank%d-----------------", rank)
            attrkey, datakey = do_in(table, rank, "key", np.int64)
            attrembedding, dataembedding = do_in(table, rank, "embedding")
            attrmomentum1, datamomentum1 = do_in(table, rank, "momentum1")
            # 打印所有返回值的形状
            logging.debug("--------------------key-----------------")
            logging.debug("Key Attributes: shape=%s, value=%s", attrkey.shape, attrkey)
            logging.debug("Key Data: shape=%s, value=%s", datakey.shape, datakey)

            logging.debug("Embedding Attributes: shape=%s, value=%s", attrembedding.shape, attrembedding)
            logging.debug("Embedding Data: shape=%s, value=%s", dataembedding.shape, dataembedding)

            logging.debug("Momentum1 Attributes: shape=%s, value=%s", attrmomentum1.shape, attrmomentum1)
            logging.debug("Momentum1 Data: shape=%s, value=%s", datamomentum1.shape, datamomentum1)
            for k, e, m in zip(datakey, dataembedding, datamomentum1):
                rankdt[k] = {"embedding": e, "momentum1": m}
            tabledt[rank] = rankdt
        embedding_dt[table] = tabledt
    return embedding_dt


def compare_structures(struct1, struct2):
    """
    递归比较两个结构（字典或元组）中的所有元素。
    """
    # 检查结构类型是否一致
    if isinstance(struct1, dict) and isinstance(struct2, dict):
        # 比较字典的键
        if sorted(struct1.keys()) != sorted(struct2.keys()):
            return False
        # 递归比较每个键对应的值
        flag = True
        for key in sorted(struct1.keys()):
            if not compare_structures(struct1[key], struct2[key]):
                logging.debug("key %s mismatch", key)
                flag = False
        return flag
    elif isinstance(struct1, np.ndarray) and isinstance(struct2, np.ndarray):
        # 比较每个numpy数组
        if not np.array_equal(struct1, struct2):
            return False
    else:
        # 其他类型直接比较
        if struct1 != struct2:
            return False

    return True


def compare_embedding_dicts(embedding_dt1, embedding_dt2):
    """
    比较两个嵌入字典，使用递归函数处理不同层次的结构。
    """
    # 检查字典长度
    if len(embedding_dt1) != len(embedding_dt2):
        raise ValueError("Embedding dictionaries have different lengths.")
    logging.debug("table_num: %d", len(embedding_dt1))

    # 递归比较整个结构
    if not compare_structures(embedding_dt1, embedding_dt2):
        logging.debug("Failed")
        return False

    logging.debug("All arrays match")
    logging.debug("Passed")
    return True


if __name__ == "__main__":
    import sys
    table1_path, table2_path, table3_path = sys.argv[1:4]
    table1 = emb_read(table1_path, table_names=["table0", "table1"], ranks=[0, 1])
    table2 = emb_read(table2_path, table_names=["table0", "table1"], ranks=[0, 1])
    table3 = emb_read(table3_path, table_names=["table0", "table1"], ranks=[0, 1])
    logging.debug("-----compare_embedding_dicts save_dir  save_dir2")
    if not compare_embedding_dicts(table1, table2):
        raise ValueError("Comparison failed between save_dir and save_dir2")
    else:
        logging.info("Comparison passed between save_dir and save_dir2")

    logging.debug("-----compare_embedding_dicts save_dir  save_dir3")
    if not compare_embedding_dicts(table2, table3):
        raise ValueError("Comparison failed between save_dir and save_dir3")
    else:
        logging.info("Comparison passed between save_dir and save_dir3")

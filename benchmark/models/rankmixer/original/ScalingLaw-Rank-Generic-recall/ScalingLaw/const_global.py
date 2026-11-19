# -*- coding: utf-8 -*-
"""
    统计相关信息
"""
import os
import torch
import pandas as pd


class DataTarget(object):
    def __init__(self, rootdir=".", filename="analysis.csv"):
        # 近30天内，数据样本总量
        self.num_all = 0
        # 近30天内，指定榜单的数据样本总量
        self.num_task_all = 0
        # 近30天内，指定榜单负采样之后的正样本数据量
        self.num_positive = 0

        # 近30天，指定榜单数据范围内的正样本比例
        self.rate_positive_task = 0.0
        # 近30天，全部榜单数据范围内的正样本比例
        self.rate_positive_all = 0.0

        self.rootdir = rootdir
        os.makedirs(rootdir, exist_ok=True)
        self.filename = filename
        self.path_save = os.path.join(self.rootdir, self.filename)

    def set_path(self, rootdir, filename):
        self.path_save = os.path.join(rootdir, filename)

    def add_num_all(self, value):
        self.num_all += value

    def add_num_task_all(self, value):
        self.num_task_all += value

    def add_num_positive(self, value):
        self.num_positive += value

    def calculate_positive_rate_task(self):
        self.rate_positive_task = self.num_positive / self.num_task_all

    def calculate_positive_rate_all(self):
        self.rate_positive_all = self.num_positive / self.num_all

    def save_data(self, rank=0):
        if rank == 0:
            num_all = self.num_all
            num_task_all = self.num_task_all
            num_positive = self.num_positive
            rate_positive_task = self.rate_positive_task
            rate_positive_all = self.rate_positive_all

            if isinstance(num_all, torch.Tensor):
                num_all = num_all.cpu().numpy()

            if isinstance(num_task_all, torch.Tensor):
                num_task_all = num_task_all.cpu().numpy()

            if isinstance(num_positive, torch.Tensor):
                num_positive = num_positive.cpu().numpy()

            if isinstance(rate_positive_task, torch.Tensor):
                rate_positive_task = rate_positive_task.cpu().numpy()

            if isinstance(rate_positive_all, torch.Tensor):
                rate_positive_all = rate_positive_all.cpu().numpy()

            dct_data = {
                "num_all": [num_all],
                "num_mask_all": [num_task_all],
                "num_positive": [num_positive],
                "rate_positive_task": [rate_positive_task],
                "rate_positive_all": [rate_positive_all],
            }

            # 保存为 CSV 文件
            df = pd.DataFrame(dct_data)
            df.to_csv(self.path_save, index=False)
            print("path_save_evaluate: ", self.path_save)


# 初始化统计对象
data_target = DataTarget()
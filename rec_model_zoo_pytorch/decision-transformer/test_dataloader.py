# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

class OfflineEnv:
    """
    Simulate an advertising bidding environment.
    """

    def __init__(self, min_remaining_budget: float = 0.1):
        """
        Initialize the simulation environment.
        :param min_remaining_budget: The minimum remaining budget allowed for bidding advertiser.
        """
        self.min_remaining_budget = min_remaining_budget

    @classmethod
    def simulate_ad_bidding(self, pvalues, pvaluesigmas, bids, leastwinningcosts):
        """
        Simulate the advertising bidding process.

        :param pValues: Values of each pv .
        :param pValueSigmas: uncertainty of each pv .
        :param bids: Bids from the bidding advertiser.
        :param leastWinningCosts: Market prices for each pv.
        :return: Win values, costs spent, and winning status for each bid.

        """
        # 确定设备（如果输入是张量）
        if isinstance(pvalues, torch.Tensor):
            device = pvalues.device

            if not isinstance(pvaluesigmas, torch.Tensor):
                pvaluesigmas = torch.tensor(pvaluesigmas, device=device, dtype=torch.float32)
            if not isinstance(bids, torch.Tensor):
                bids = torch.tensor(bids, device=device, dtype=torch.float32)
            if not isinstance(leastwinningcosts, torch.Tensor):
                leastwinningcosts = torch.tensor(leastwinningcosts, device=device, dtype=torch.float32)

            tick_status = bids >= leastwinningcosts
            tick_cost = leastwinningcosts * tick_status.float()

            values = torch.normal(mean=pvalues, std=pvaluesigmas)
            values = values * tick_status.float()
            tick_value = torch.clamp(values, min=0, max=1)

            tick_conversion = torch.bernoulli(tick_value)

        else:
            tick_status = bids >= leastwinningcosts
            tick_cost = leastwinningcosts * tick_status
            values = np.random.normal(loc=pvalues, scale=pvaluesigmas)
            values = values * tick_status
            tick_value = np.clip(values, 0, 1)
            tick_conversion = np.random.binomial(n=1, p=tick_value)

        return tick_value, tick_cost, tick_status, tick_conversion


class TestDataLoader:
    """
    Offline evaluation data loader.
    """

    def __init__(self, file_path="./data/log.csv"):
        """
        Initialize the data loader.
        Args: file_path (str): The path to the training data file.
        """
        
        self.file_path = file_path
        self.raw_data_path = os.path.join(os.path.dirname(file_path), "raw_data.pickle")
        self.raw_data = self._get_raw_data()
        self.keys, self.test_dict = self._get_test_data_dict()

    def mock_data(self, key):
        """
        Get training data based on deliveryPeriodIndex and advertiserNumber, and construct the test data.
        """
        data = self.test_dict[key]
        pvalues = data.groupby('timeStepIndex')['pValue'].apply(list).apply(np.array).tolist()
        pvaluesigmas = data.groupby('timeStepIndex')['pValueSigma'].apply(list).apply(np.array).tolist()
        leastwinningcosts = data.groupby('timeStepIndex')['leastWinningCost'].apply(list).apply(np.array).tolist()
        num_timestepindex = len(pvalues)
        return num_timestepindex, pvalues, pvaluesigmas, leastwinningcosts

    def _get_raw_data(self):
        """
        Read raw data from a pickle file.
        Returns:pd.DataFrame: The raw data as a DataFrame.
        """

        if os.path.exists(self.raw_data_path):
            with open(self.raw_data_path, 'rb') as file:
                return pickle.load(file)
        else:
            tem = pd.read_csv(self.file_path)
            with open(self.raw_data_path, 'wb') as file:
                pickle.dump(tem, file)
            return tem

    def _get_test_data_dict(self):
        """
        Group and sort the raw data by deliveryPeriodIndex and advertiserNumber.
        Returns:
            list: A list of group keys.
            dict: A dictionary with grouped data.
        """
        grouped_data = self.raw_data.sort_values('timeStepIndex').groupby(['deliveryPeriodIndex', 'advertiserNumber'])
        data_dict = {key: group for key, group in grouped_data}
        return list(data_dict.keys()), data_dict

    
if __name__ == '__main__':
    pass
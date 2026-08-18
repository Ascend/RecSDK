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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
import torch

@dataclass
class BiddingHistory:
    history_pvalue_info: Any
    history_bid: Any
    history_auction_result: Any
    history_impression_result: Any
    history_least_winning_cost: Any


@dataclass
class BiddingInputs:
    time_step_index: int
    p_values: Any
    p_value_sigmas: Any
    history: BiddingHistory


class BaseBiddingStrategy(ABC):
    """
    Base bidding strategy interface defining methods to be implemented.
    """

    def __init__(self, budget=100, name="BaseStrategy", cpa=2, category=1):
        """
        Initialize the bidding strategy.
        parameters:
            @budget: the advertiser's budget for a delivery period.
            @cpa: the CPA constraint of the advertiser.
            @category: the index of advertiser's industry category.
        """
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.cpa = cpa
        self.category = category

    @abstractmethod
    def reset(self):
        """
        Reset the remaining budget to its initial state.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def bidding(self, inputs: BiddingInputs):
        """
        Bids for all the opportunities in a delivery period

        inputs fields:
         @time_step_index: the index of the current decision time step.
         @p_values: the conversion action probability.
         @p_value_sigmas: the prediction probability uncertainty.
         @history: contains history_* fields (pvalue_info, bid, auction_result, impression_result, least_winning_cost)

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        pass


class MyBiddingStrategy(BaseBiddingStrategy):
    """
    Decision-Transformer-PlayerStrategy with device support
    """

    def __init__(self, params, model=None, budget=100, name="Decision-Transformer-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        self.device = params.device
        self.params = params

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, inputs: BiddingInputs):
        device = self.device

        time_step_index = inputs.time_step_index
        p_values = inputs.p_values
        p_value_sigmas = inputs.p_value_sigmas  

        history_pvalue_info = inputs.history.history_pvalue_info
        history_bid = inputs.history.history_bid
        history_auction_result = inputs.history.history_auction_result
        history_impression_result = inputs.history.history_impression_result
        history_least_winning_cost = inputs.history.history_least_winning_cost

        time_left = torch.tensor((48 - time_step_index) / 48, device=device, dtype=torch.float32)
        budget_left = (
            torch.tensor(self.remaining_budget / self.budget, device=device, dtype=torch.float32)
            if self.budget > 0
            else torch.tensor(0.0, device=device)
        )

        history_xi = [
            torch.as_tensor(result, device=device, dtype=torch.float32)[:, 0]
            for result in history_auction_result
        ]

        history_pvalue = [
            torch.as_tensor(result, device=device, dtype=torch.float32)[:, 0]
            for result in history_pvalue_info
        ]

        history_conversion = [
            torch.as_tensor(result, device=device, dtype=torch.float32)[:, 1]
            for result in history_impression_result
        ]

        history_least_winning_cost = [
            torch.as_tensor(price, device=device, dtype=torch.float32)
            for price in history_least_winning_cost
        ]

        history_bid = [
            torch.as_tensor(bid, device=device, dtype=torch.float32)
            for bid in history_bid
        ]

        def mean_of_means(tensor_list):
            if len(tensor_list) == 0:
                return torch.tensor(0.0, device=device)
            return torch.stack([t.mean() for t in tensor_list]).mean()

        historical_xi_mean = mean_of_means(history_xi)
        historical_conversion_mean = mean_of_means(history_conversion)
        historical_least_winning_cost_mean = mean_of_means(history_least_winning_cost)
        historical_pvalues_mean = mean_of_means(history_pvalue)
        historical_bid_mean = mean_of_means(history_bid)

        def mean_of_last_n_elements(history, n):
            last_data = history[max(0, n - 3):n]
            if len(last_data) == 0:
                return torch.tensor(0.0, device=device)
            return torch.stack([t.mean() for t in last_data]).mean()

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_least_winning_cost_mean = mean_of_last_n_elements(history_least_winning_cost, 3)
        last_three_pvalues_mean = mean_of_last_n_elements(history_pvalue, 3)
        last_three_bid_mean = mean_of_last_n_elements(history_bid, 3)

        current_pvalues = torch.tensor(p_values, device=self.device, dtype=torch.float32)
        current_pvalues_mean = current_pvalues.mean()
        current_pv_num = torch.tensor(current_pvalues.numel(), dtype=torch.float32, device=device)

        historical_pv_num_total = (
            torch.tensor(sum(b.numel() for b in history_bid), device=device)
            if len(history_bid) > 0
            else torch.tensor(0.0, device=device)
        )

        last_three_pv_num_total = (
            torch.tensor(
                sum(history_bid[i].numel()
                    for i in range(max(0, time_step_index - 3), time_step_index)),
                device=device
            )
            if len(history_bid) > 0
            else torch.tensor(0.0, device=device)
        )

        test_state = torch.tensor([
            time_left, budget_left,
            historical_bid_mean, last_three_bid_mean,
            historical_least_winning_cost_mean, historical_pvalues_mean,
            historical_conversion_mean, historical_xi_mean,
            last_three_least_winning_cost_mean, last_three_pvalues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pvalues_mean, current_pv_num,
            last_three_pv_num_total, historical_pv_num_total
        ], device=self.device, dtype=torch.float32)

        pre_reward = (
            history_conversion[-1].sum()
            if len(history_conversion) > 0
            else None
        )
        return test_state, pre_reward, current_pvalues
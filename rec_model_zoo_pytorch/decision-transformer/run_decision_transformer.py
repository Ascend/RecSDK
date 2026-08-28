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

import sys
import os

root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
sys.path.append(root_path)

import math
import pickle
import time
from dataclasses import fields, is_dataclass
import torch
import numpy as np

from easydict import EasyDict as edict
from torch.utils.data import DataLoader, WeightedRandomSampler

from dt import DecisionTransformer, DTInputs, DTBatch 
from common import Profiler, get_params, get_opts, output_report
from bidding_strategy import MyBiddingStrategy, BiddingInputs, BiddingHistory
from set_env import EpisodeReplayBuffer, save_normalize_dict, tensor_to_numpy, getscore_neurips, logger
from test_dataloader import TestDataLoader, OfflineEnv
from check_result import IoChecker

class Rundt():
    def __init__(self, params):
        self.params = params
        self.device = params.device
        self.data_path = params.data_dir
        self.test_dir = params.test_dir
        self.state_dim = params.state_dim
        self.batch_size = params.batch_size
        self.step_num = params.step_num
        self.test_path = self.params.test_dir
        self.model = None
        self._shape_list = []
        self._shape_idx = 0
        self._dynamic_marked = False
        self._compile_count = 0

        shape_list_str = os.getenv("SHAPE_LIST", "").strip()
        if self.params.enable_dynamic_compile is None and shape_list_str:
            for pair in shape_list_str.split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                try:
                    values = [value.strip() for value in pair.split(",")]
                    if len(values) != 2:
                        raise ValueError
                    batch_size, seq_len = (int(value) for value in values)
                except ValueError as exc:
                    raise ValueError(
                        "SHAPE_LIST entries must use the batch_size,seq_len format"
                    ) from exc
                if batch_size < 1 or seq_len < 1:
                    raise ValueError(
                        "SHAPE_LIST entries must contain positive batch_size and seq_len"
                    )
                if seq_len > self.params.k:
                    raise ValueError(
                        f"SHAPE_LIST seq_len must not exceed model context length k={self.params.k}"
                    )
                self._shape_list.append((batch_size, seq_len))
        self.dynamic_enabled = bool(self._shape_list)
        self._dynamic_batch = len({batch_size for batch_size, _ in self._shape_list}) > 1
        self._dynamic_seq_len = len({seq_len for _, seq_len in self._shape_list}) > 1

    def _next_shape(self):
        shape = self._shape_list[self._shape_idx % len(self._shape_list)]
        self._shape_idx += 1
        return shape

    def _generate_inputs(self, batch_size, seq_len):
        device = self.device
        return DTInputs(
            states=torch.randn(batch_size, seq_len, self.params.state_dim, device=device),
            actions=torch.randn(batch_size, seq_len, self.params.act_dim, device=device),
            rewards=torch.randn(batch_size, seq_len, 1, device=device),
            returns_to_go=torch.randn(batch_size, seq_len, 1, device=device),
            timesteps=torch.arange(seq_len, device=device)
            .expand(batch_size, -1)
            .contiguous(),
            attention_mask=torch.ones(
                batch_size, seq_len, dtype=torch.long, device=device
            ),
        )

    def _mark_dynamic_inputs(
        self, value, mark_batch=False, mark_seq_len=False, field_name=None
    ):
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            if mark_batch:
                torch._dynamo.mark_dynamic(value, 0)
            if (
                mark_seq_len
                and field_name in {
                    "states",
                    "actions",
                    "rewards",
                    "returns_to_go",
                    "timesteps",
                    "attention_mask",
                }
                and value.ndim >= 2
            ):
                torch._dynamo.mark_dynamic(value, 1)
        elif isinstance(value, dict):
            for name, item in value.items():
                self._mark_dynamic_inputs(
                    item,
                    mark_batch=mark_batch,
                    mark_seq_len=mark_seq_len,
                    field_name=name,
                )
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                self._mark_dynamic_inputs(
                    getattr(value, field.name),
                    mark_batch=mark_batch,
                    mark_seq_len=mark_seq_len,
                    field_name=field.name,
                )
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._mark_dynamic_inputs(
                    item,
                    mark_batch=mark_batch,
                    mark_seq_len=mark_seq_len,
                    field_name=field_name,
                )

    def _register_compile_callbacks(self):
        def on_compile_start(*_args, **_kwargs):
            self._compile_count += 1
            logger.info(f"[compile {self._compile_count}] start")

        def on_compile_end(*_args, **_kwargs):
            logger.info(f"[compile {self._compile_count}] end")

        if hasattr(torch._dynamo, "on_compile_start"):
            torch._dynamo.on_compile_start(on_compile_start)
        if hasattr(torch._dynamo, "on_compile_end"):
            torch._dynamo.on_compile_end(on_compile_end)

    def _run_dynamic_shape_benchmark(self):
        self._shape_idx = 0
        warmup_count = len(self._shape_list)
        iteration_count = max(self.step_num, len(self._shape_list))
        times_range = []
        batches_list = []
        profiler = Profiler(self.params)
        profiler.cur_batch_size = "Dynamic"

        with torch.no_grad():
            # Compiling exactly at the fixed context boundary can make some
            # backends specialize the causal-mask slice to k. Seed dynamic
            # compilation with a shorter listed sequence when necessary.
            if (
                self._dynamic_seq_len
                and self._shape_list[0][1] == self.params.k
            ):
                seed_shape = next(
                    (
                        shape
                        for shape in self._shape_list
                        if shape[1] < self.params.k
                    ),
                    None,
                )
                if seed_shape is not None:
                    seed_inputs = self._generate_inputs(*seed_shape)
                    self._mark_dynamic_inputs(
                        seed_inputs,
                        mark_batch=self._dynamic_batch,
                        mark_seq_len=self._dynamic_seq_len,
                    )
                    self._dynamic_marked = True
                    logger.info(
                        f"[compile seed] shape=({seed_shape[0]}, {seed_shape[1]})"
                    )
                    self.model(seed_inputs)

            for index in range(warmup_count):
                batch_size, seq_len = self._next_shape()
                inputs = self._generate_inputs(batch_size, seq_len)
                if not self._dynamic_marked:
                    self._mark_dynamic_inputs(
                        inputs,
                        mark_batch=self._dynamic_batch,
                        mark_seq_len=self._dynamic_seq_len,
                    )
                    logger.info(
                        "Dynamic dimensions: "
                        f"batch={self._dynamic_batch}, seq_len={self._dynamic_seq_len}"
                    )
                    self._dynamic_marked = True
                logger.info(f"[warmup {index}] shape=({batch_size}, {seq_len})")
                self.model(inputs)

            # Keep compilation and warmup outside profiling. The existing
            # profiler schedule is applied only to steady-state iterations.
            with profiler.get_profiler() as prof:
                for index in range(iteration_count):
                    batch_size, seq_len = self._next_shape()
                    inputs = self._generate_inputs(batch_size, seq_len)
                    logger.info(f"[iter {index}] shape=({batch_size}, {seq_len})")
                    self.model.synchronize()
                    start_time = time.time()
                    self.model(inputs)
                    self.model.synchronize()
                    times_range.append(time.time() - start_time)
                    batches_list.append(batch_size)
                    if self.params.profiling_mode and hasattr(prof, "step"):
                        prof.step()

        output_report(
            times_range,
            batches_list,
            graph_=self.params.graph,
            compile_=self.params.compile,
        )
        logger.info(
            f"Dynamic Shape benchmark completed: compile_count={self._compile_count}, "
            f"iterations={iteration_count}, qps={sum(batches_list) / sum(times_range):.2f}"
        )

    def train(self):
        replay_buffer = EpisodeReplayBuffer(16, 1, self.data_path, self.params)
        save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                            self.params.modelsave_dir)
        logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

        self.model = DecisionTransformer(state_mean=replay_buffer.state_mean, state_std=replay_buffer.state_std,
                                         params=params)
        self.model.to(self.device)
        self.model.device = self.device
        self.model.set_hf32()

        sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=self.step_num * self.batch_size,
                                        replacement=True)
        dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=self.batch_size)

        self.model.train()

        profiler = Profiler(params)
        profiling = profiler.get_profiler()
        with profiling as prof:
            for i, batch in enumerate(dataloader, start=1):
                (states, actions, rewards, dones, rtg, timesteps, attention_mask) = batch
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                rtg = rtg.to(self.device)
                timesteps = timesteps.to(self.device)
                attention_mask = attention_mask.to(self.device)

                batch = DTBatch(
                        states=states,              
                        actions=actions,           
                        rewards=rewards,            
                        rtg=rtg,                 
                        timesteps=timesteps,       
                        attention_mask=attention_mask, 
                    )
                train_loss = self.model.step(batch)

                logger.info(f"[Step {i}] Loss: {train_loss:.4f}")
                self.model.scheduler.step()
                if "cuda" in self.params.device:
                    torch.cuda.synchronize()
                elif "npu" in self.params.device:
                    torch.npu.synchronize()
                prof.step()

        self.model.save_net()      


    def eval(self):
        replay_buffer = EpisodeReplayBuffer(16, 1, self.data_path, self.params)

        logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

        self.model = DecisionTransformer(state_mean=replay_buffer.state_mean, state_std=replay_buffer.state_std,
                                         params=params)
        self.model.to(self.device)
        self.model.device = self.device
        self.fp_32(self.params)
        self.set_compile_model()

        sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=self.step_num * self.batch_size,
                                        replacement=True)
        dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=self.batch_size)

        profiler = Profiler(params)
        profiling = profiler.get_profiler()

        current_step = 0
        with torch.no_grad():  # 禁用梯度
            with profiling as prof:
                for batch in dataloader:
                    current_step += 1
                    (states, actions, rewards, dones, rtg, timesteps, attention_mask) = batch

                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    rtg = rtg.to(self.device)
                    timesteps = timesteps.to(self.device)
                    attention_mask = attention_mask.to(self.device)

                    states = states.float()
                    actions = actions.float()
                    rewards = rewards.float()
                    rtg = rtg.float()

                    state_preds, action_preds, return_preds, reward_preds = self.model.forward(
                        states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask
                    )
                    if "cuda" in self.params.device:
                        torch.cuda.synchronize()
                    elif "npu" in self.params.device:
                        torch.npu.synchronize()
                    prof.step()
                    logger.info(f"[Record Step {current_step}] Profiling forward pass")

            logger.info("Profiling completed, start parsing data...")
            if hasattr(profiler, "parse"):
                profiler.parse()
            logger.info("Profiling data parsed successfully!")


    def test(self):
        device = self.params.device

        model_path = os.path.join(self.params.modelsave_dir, "dt.pt")
        pickle_path = os.path.join(self.params.modelsave_dir, "normalize_dict.pkl")

        with open(pickle_path, 'rb') as f:
            normalize_dict = pickle.load(f)

        self.model = DecisionTransformer(state_mean=normalize_dict["state_mean"],
                                         state_std=normalize_dict["state_std"],
                                         params=self.params).to(self.device)

        self.model.load_net(model_path=model_path)
        self.model.set_hf32()
        if self.dynamic_enabled:
            self._register_compile_callbacks()
        self.model.set_compile_model()
        self.model.eval()

        if self.dynamic_enabled:
            self._run_dynamic_shape_benchmark()
            return

        data_loader = TestDataLoader(self.test_path)
        env = OfflineEnv()
        agent = MyBiddingStrategy(self.params, model=self.model)

        keys, test_dict = data_loader.keys, data_loader.test_dict
        key = keys[0]

        num_timestepindex, pvalues, pvaluesigmas, leastwinningcosts = data_loader.mock_data(key)
        rewards = torch.zeros(num_timestepindex, device=device, dtype=torch.float32)
        history = {
            'historybids': [],
            'historyauctionresult': [],
            'historyimpressionresult': [],
            'historyleastwinningcost': [],
            'historypvalueinfo': []
        }

        profiler = Profiler(params)
        profiling = profiler.get_profiler()
        times_range = []
        batch_size = []
        checker = IoChecker(params)

        with profiling as prof:
            for timestep_index in range(num_timestepindex):
                logger.info(f'Timestep Index: {timestep_index + 1} Begin')
                start_time = time.time()
                pvalue = torch.tensor(pvalues[timestep_index], device=device, dtype=torch.float32)
                pvaluesigma = torch.tensor(pvaluesigmas[timestep_index], device=device, dtype=torch.float32)
                leastwinningcost = torch.tensor(leastwinningcosts[timestep_index], device=device, dtype=torch.float32)

                if agent.remaining_budget < env.min_remaining_budget:
                    bid = torch.zeros(pvalue.shape[0], device=device)
                else:
                    inputs = BiddingInputs(
                        time_step_index=timestep_index,
                        p_values=pvalue,
                        p_value_sigmas=pvaluesigma,
                        history=BiddingHistory(
                            history_pvalue_info=history["historypvalueinfo"],
                            history_bid=history["historybids"],
                            history_auction_result=history["historyauctionresult"],
                            history_impression_result=history["historyimpressionresult"],
                            history_least_winning_cost=history["historyleastwinningcost"],
                        )
                    )
                    test_state, pre_reward, current_pvalues = agent.bidding(inputs)

                    inputs = {
                        "test_state": test_state,
                        "pre_reward": pre_reward,
                    }
                    model_inputs = checker.load_or_save_inputs(inputs, timestep_index)

                    alpha = self.model.take_actions(test_state, pre_reward=pre_reward)

                    checker.save_outputs(alpha, timestep_index)
                    bid = alpha * current_pvalues

                # 模拟广告竞价，确保 env.simulate_ad_bidding 返回张量
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                    pvalue, pvaluesigma, bid, leastwinningcost
                )

                # 确保所有操作在设备上
                over_cost_ratio = torch.clamp(
                    (torch.sum(tick_cost) - agent.remaining_budget) / (torch.sum(tick_cost) + 1e-4),
                    min=0
                )

                while over_cost_ratio > 0:
                    pv_index = torch.where(tick_status == 1)[0]
                    num_to_drop = int(math.ceil(pv_index.shape[0] * over_cost_ratio.item()))
                    if num_to_drop > 0:
                        dropped_pv_index = pv_index[torch.randperm(pv_index.shape[0])[:num_to_drop]]
                        bid[dropped_pv_index] = 0
                        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                            pvalue, pvaluesigma, bid, leastwinningcost
                        )
                        over_cost_ratio = torch.clamp(
                            (torch.sum(tick_cost) - agent.remaining_budget) / (torch.sum(tick_cost) + 1e-4),
                            min=0
                        )
                    else:
                        break
                agent.remaining_budget -= torch.sum(tick_cost)
                rewards[timestep_index] = torch.sum(tick_conversion)

                # 存储历史数据
                temhistorypvalueinfo = torch.stack([pvalue, pvaluesigma], dim=1).detach()  # [num_pv, 2]
                history["historypvalueinfo"].append(temhistorypvalueinfo)
                history["historybids"].append(bid.detach())
                history["historyleastwinningcost"].append(leastwinningcost.detach())

                # 创建拍卖结果张量
                temauctionresult = torch.stack([tick_status, tick_status, tick_cost], dim=1).detach()
                history["historyauctionresult"].append(temauctionresult)

                temimpressionresult = torch.stack([tick_conversion, tick_conversion], dim=1).detach()
                history["historyimpressionresult"].append(temimpressionresult)
                if "cuda" in self.params.device:
                    torch.cuda.synchronize()
                elif "npu" in self.params.device:
                    torch.npu.synchronize()
                end_time = time.time()
                if timestep_index == 25:
                    if "cuda" in self.params.device:
                        times_range.append(end_time - start_time)
                        batch_size.append(timestep_index)
                if timestep_index > 25:
                    if "npu" in self.params.device:
                        times_range.append(end_time - start_time)
                        batch_size.append(timestep_index)
                prof.step()
                logger.info(f'Timestep Index: {timestep_index + 1} End')
        output_report(times_range, batch_size, graph_=self.params.graph, compile_=self.params.compile)

        all_reward = torch.sum(rewards).item()
        all_cost = agent.budget - agent.remaining_budget.item()
        cpa_real = all_cost / (all_reward + 1e-10)
        cpa_constraint = agent.cpa
        score = getscore_neurips(all_reward, cpa_real, cpa_constraint)

        logger.info(f'Total Reward: {all_reward}')
        logger.info(f'Total Cost: {all_cost}')
        logger.info(f'CPA-real: {cpa_real}')
        logger.info(f'CPA-constraint: {cpa_constraint}')
        logger.info(f'Score: {score}')

if __name__ == "__main__":
    params = get_params()
    params.update(
        edict(
            {
            "length_times": 3,
            "hidden_size": 2048,
            "state_dim": 16,
            "act_dim": 1,
            "k": 64,
            "max_ep_len": 96,
            "scale": 2000,
            "target_return": 4,
            "model": "dt",
            }
        )
    )
    params = get_opts(sys.argv, params)
    handler = Rundt(params)
    if params.mode == 'train':
        handler.train()
    elif params.mode == 'eval':
        handler.eval()
    else:
        handler.test()

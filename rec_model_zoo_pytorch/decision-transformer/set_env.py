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

import ast
import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("npu").numpy()
    return x

def getscore_neurips(reward, cpa, constraint):
    beta = 2
    penalty = 1
    if cpa > constraint:
        coef = constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward

def _get_logger(log_level: str) -> logging.Logger:
    model_zoo_logger = logging.getLogger("rec_model_zoo_pytorch")
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S %p"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    model_zoo_logger.addHandler(stream_handler)
    model_zoo_logger.setLevel(log_level)
    return model_zoo_logger

logger = _get_logger("INFO")

def save_normalize_dict(normalize_dict, save_dir):
    """
    Save the normalization dictionary to a Pickle file.

    Args:
        normalize_dict: The dictionary containing normalization statistics.
        save_dir: The directory to save the normalization dictionary.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'normalize_dict.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(normalize_dict, file)

class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path, params, max_eplen=24, return_scale=2000, context_len=20):
        self.device = params.device
        super(EpisodeReplayBuffer, self).__init__()
        self.max_ep_len = max_eplen
        self.scale = return_scale

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_len = context_len
        training_data = pd.read_csv(data_path)

        def safe_literal_eval(val):
            if pd.isna(val):
                return val
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return val
        training_data["state"] = training_data["state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
        self.trajectories = training_data

        self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones = [], [], [], [], [], []
        state = []
        reward = []
        action = []
        dones = []
        for _, row in self.trajectories.iterrows():
            state.append(row["state"])
            reward.append(row['reward'])
            action.append(row["action"])
            dones.append(row["done"])
            if row["done"]:
                if len(state) != 1:
                    self.states.append(np.array(state))
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    self.dones.append(np.array(dones))
                state = []
                reward = []
                action = []
                dones = []
        self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns)

        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0) + 1e-6

        self.trajectories = []
        for i, state in enumerate(self.states):
            self.trajectories.append({
                "observations": state,
                "actions": self.actions[i],
                "rewards": self.rewards[i],
                "dones": self.dones[i]
            })


        self.pct_traj = 1.

        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]

        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])

    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = random.randint(0, traj['rewards'].shape[0] - 1)

        s = traj['observations'][start_t: start_t + self.context_len]
        a = traj['actions'][start_t: start_t + self.context_len]
        r = traj['rewards'][start_t: start_t + self.context_len].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.context_len]
        else:
            d = traj['dones'][start_t: start_t + self.context_len]
        timesteps = np.arange(start_t, start_t + s.shape[0])
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
        rtg = self.discount_cumsum(traj['rewards'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        tlen = s.shape[0]
        s = np.concatenate([np.zeros((self.context_len - tlen, self.state_dim)), s], axis=0)
        s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((self.context_len - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.context_len - tlen, 1)), r], axis=0)
        r = r / self.scale
        d = np.concatenate([np.ones((self.context_len - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.context_len - tlen, 1)), rtg], axis=0) / self.scale
        timesteps = np.concatenate([np.zeros((self.context_len - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.context_len- tlen)), np.ones((tlen))], axis=0)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        return s, a, r, d, rtg, timesteps, mask

    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
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

import math
import os
import threading

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_env import logger


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['n_embd'] % config['n_head'] != 0:
            raise ValueError(
                f"n_embd ({config['n_embd']}) must be divisible by n_head ({config['n_head']})"
            )
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])

        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])

        self.register_buffer("bias",
                             torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'],
                                                                                           config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']
        self._attn_map = None

    def forward(self, x, mask):
        batch_size, seq_len, d_model = x.size()
        n_heads = self.n_head
        d_head = d_model // n_heads

        k = self.key(x).view(batch_size, seq_len, self.n_head, d_head).contiguous().transpose(1, 2)
        q = self.query(x).view(batch_size, seq_len, self.n_head, d_head).contiguous().transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_head, d_head).contiguous().transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1))
        att = att * (1.0 / math.sqrt(d_head))
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        att = att.masked_fill(causal_mask == 0, -1e4)

        # padding mask
        if mask is not None:
            pad_mask = mask[:, None, None, :]  # [B,1,1,T]
            att = att.masked_fill(pad_mask == 0, -1e4)

        att = F.softmax(att, dim=-1)
        self._attn_map = att.detach()  

        att = self.attn_drop(att)
        y = torch.matmul(att, v)

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Linear(config['n_inner'], config['n_embd']),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, inputs_embeds, attention_mask):
        attn_output = self.attn(self.ln1(inputs_embeds), attention_mask)
        x = inputs_embeds + attn_output
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output
        return x

@dataclass
class DTInputs:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    returns_to_go: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


@dataclass
class DTBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    rtg: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: torch.Tensor

    def to_inputs(self) -> DTInputs:
        return DTInputs(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            returns_to_go=self.rtg[:, :-1],  
            timesteps=self.timesteps,
            attention_mask=self.attention_mask,
        )


class DecisionTransformer(nn.Module):
    def __init__(self, state_mean, state_std, params, action_tanh=False, ):
        super(DecisionTransformer, self).__init__()
        self.params = params
        self.device = params.device
        self.state_dim = params.state_dim
        self.act_dim = params.act_dim
        self.max_length = params.k
        self.max_ep_len = params.max_ep_len
        self.scale = params.scale
        self.target_return = params.target_return
        self.length_times = params.length_times
        self.hidden_size = params.hidden_size

        self.hf32 = params.hf32
        self.compile = params.compile
        self.enable_dynamic_compile = params.enable_dynamic_compile
        self.graph = params.graph
        self.shape_handle = params.shape_handle
        self.graphs = {}

        if isinstance(state_mean, torch.Tensor):
            self.state_mean = state_mean.to(self.device)
        else:
            self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self.device)

        if isinstance(state_std, torch.Tensor):
            self.state_std = state_std.to(self.device)
        else:
            self.state_std = torch.tensor(state_std, dtype=torch.float32, device=self.device)

        self.warmup_steps = 10
        self.weight_decay = 0.0001
        self.learning_rate = 0.0001
        self.save_path = params.modelsave_dir

        block_config = {
            "n_ctx": 3 * self.max_length,
            "n_embd": self.hidden_size,
            "n_layer": 8,
            "n_head": 16,
            "n_inner": 4096,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1
        }

        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(self.hidden_size, 1)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.to(self.device)
        self.init_eval()

    def forward(self, inputs: DTInputs):
        states = inputs.states
        actions = inputs.actions
        rewards = inputs.rewards
        returns_to_go = inputs.returns_to_go
        timesteps = inputs.timesteps
        attention_mask = inputs.attention_mask

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        if timesteps.dtype != torch.long:
            timesteps = timesteps.long()
        timesteps = timesteps.clamp(0, self.max_ep_len - 1)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings.add_(time_embeddings)
        action_embeddings.add_(time_embeddings)
        returns_embeddings.add_(time_embeddings)
        rewards_embeddings.add_(time_embeddings)

        stacked_inputs = torch.cat(
            (returns_embeddings, state_embeddings, action_embeddings), 
            dim=1
        )
        stacked_inputs = stacked_inputs.reshape(batch_size, 3 * seq_length, self.hidden_size).contiguous()

        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = attention_mask.unsqueeze(1).expand(-1, 3, -1)
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, 3 * seq_length)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        x = x.view(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])
        dummy = torch.empty(0, device=states.device)

        return [state_preds, action_preds, return_preds, dummy]



    def get_action(self, inputs: DTInputs, **kwargs):
        device = self.device
        with torch.no_grad():
            states = inputs.states.reshape(1, -1, self.state_dim)
            actions = inputs.actions.reshape(1, -1, self.act_dim)
            returns_to_go = inputs.returns_to_go.reshape(1, -1, 1)
            rewards = inputs.rewards.reshape(1, -1, 1)
            timesteps = inputs.timesteps.reshape(1, -1)

            attention_mask = None
            if self.max_length is not None and states.shape[1] < self.max_length:
                # 预先计算填充长度
                pad_len = self.max_length - states.shape[1]

                # 创建填充张量
                state_pad = torch.zeros((1, pad_len, self.state_dim), device=device)
                action_pad = torch.zeros((1, pad_len, self.act_dim), device=device)
                return_pad = torch.zeros((1, pad_len, 1), device=device)
                reward_pad = torch.zeros((1, pad_len, 1), device=device)
                timestep_pad = torch.zeros((1, pad_len), device=device, dtype=torch.long)

                # 组合填充
                states = torch.cat([state_pad, states], dim=1)
                actions = torch.cat([action_pad, actions], dim=1)
                returns_to_go = torch.cat([return_pad, returns_to_go], dim=1)
                rewards = torch.cat([reward_pad, rewards], dim=1)
                timesteps = torch.cat([timestep_pad, timesteps], dim=1)

                # 创建mask
                attention_mask = torch.zeros(1, self.max_length, device=device, dtype=torch.long)
                attention_mask[0, -states.shape[1]:] = 1
            else:
                attention_mask = None
                # 如果超过max_length，截断
                if self.max_length is not None and states.shape[1] > self.max_length:
                    states = states[:, -self.max_length:]
                    actions = actions[:, -self.max_length:]
                    returns_to_go = returns_to_go[:, -self.max_length:]
                    rewards = rewards[:, -self.max_length:]
                    timesteps = timesteps[:, -self.max_length:]
                    attention_mask = torch.ones(1, self.max_length, device=device, dtype=torch.long)

            inputs = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "returns_to_go": returns_to_go,
                "timesteps": timesteps,
                "attention_mask": attention_mask
            }

            if self.is_manual_graph():
                _, action_preds, _, _ = self.model_infer_graph(inputs, batch_size=1)
            else:
                dt_inputs = DTInputs(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        returns_to_go=returns_to_go,
                        timesteps=timesteps,
                        attention_mask=attention_mask,
                    )
                _, action_preds, _, _ = self.forward(dt_inputs)
            return action_preds[0, -1]


    def take_actions(self, state, target_return=None, pre_reward=None):
        """
        Device-safe, profiling-safe DT action inference
        """
        self.eval()
        device = self.device
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                state_tensor = state.to(device=device, dtype=torch.float32).reshape(1, -1)
            else:
                state_tensor = torch.from_numpy(state).to(device=device, dtype=torch.float32).reshape(1, -1)

            if self.eval_states is None:
                self.eval_states = state_tensor

                ep_return = target_return if target_return is not None else self.target_return
                self.eval_target_return = torch.tensor(
                    ep_return, dtype=torch.float32, device=device
                ).reshape(1, 1)

                self.eval_actions = torch.zeros((0, self.act_dim), device=device)
                self.eval_rewards = torch.zeros((0,), device=device)
                self.eval_timesteps = torch.zeros((1, 1), dtype=torch.long, device=device)

            else:
                #pre_reward 
                if pre_reward is None:
                    raise ValueError("pre_reward must be provided after first timestep")

                if not isinstance(pre_reward, torch.Tensor):
                    pre_reward = torch.tensor(pre_reward, dtype=torch.float32, device=device)
                else:
                    pre_reward = pre_reward.to(device)

                if len(self.eval_rewards) > 0:
                    self.eval_rewards = self.eval_rewards[:-1] 
                    self.eval_rewards = torch.cat([self.eval_rewards, pre_reward.unsqueeze(0)], dim=0)

                self.eval_states = torch.cat([self.eval_states, state_tensor], dim=0)

                pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
                self.eval_target_return = torch.cat(
                    [self.eval_target_return, pred_return.view(1, 1)], dim=1
                )

                self.eval_timesteps = torch.cat(
                    [
                        self.eval_timesteps,
                        (self.eval_timesteps[:, -1:] + 1),
                    ],
                    dim=1,
                )
            self.eval_actions = torch.cat(
                [self.eval_actions, torch.zeros((1, self.act_dim), device=device, dtype=torch.float32)], dim=0
            )
            self.eval_rewards = torch.cat(
                [self.eval_rewards, torch.zeros((1,), device=device, )], dim=0
            )
            states = (self.eval_states - self.state_mean) / self.state_std

            infer_inputs = DTInputs(
                states=states.unsqueeze(0),                       
                actions=self.eval_actions.unsqueeze(0),           
                rewards=self.eval_rewards.view(1, -1, 1),         
                returns_to_go=self.eval_target_return.view(1, -1, 1),  
                timesteps=self.eval_timesteps,                    
                attention_mask=None,                              
            )
            action = self.get_action(infer_inputs)

            self.eval_actions[-1] = action
            return action


    def step(self, 
        batch: DTBatch,
        state_preds=None,
        action_preds=None,
        return_preds=None,
        reward_preds=None,):

        rewards_target = torch.clone(batch.rewards)
        action_target = torch.clone(batch.actions)
        rtg_target = torch.clone(batch.rtg)

        predictions_provided = [state_preds, action_preds, return_preds, reward_preds]
        all_preds_given = all(pred is not None for pred in predictions_provided)

        if not all_preds_given:
            try:
                forward_outputs = self.forward(batch.to_inputs())
            except (RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "DecisionTransformer forward pass failed during training step"
                ) from exc
            if not isinstance(forward_outputs, (list, tuple)) or len(forward_outputs) != 4:
                raise RuntimeError(
                    "DecisionTransformer.forward must return four prediction tensors"
                )
            state_preds, action_preds, return_preds, reward_preds = forward_outputs

        act_dim = action_preds.shape[2]
        mask = batch.attention_mask.reshape(-1) > 0
        action_preds = action_preds.reshape(-1, act_dim)[mask]
        action_target = action_target.reshape(-1, act_dim)[mask]

        loss = torch.mean((action_preds - action_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()
        return loss

    def loss(self, batch: DTBatch):
        action_target = torch.clone(batch.actions)
        _, action_preds, _, _ = self.forward(batch.to_inputs())

        act_dim = action_preds.shape[2]
        mask = batch.attention_mask.reshape(-1) > 0
        action_preds = action_preds.reshape(-1, act_dim)[mask]
        action_target = action_target.reshape(-1, act_dim)[mask]

        loss = torch.mean((action_preds - action_target) ** 2)
        return loss


    def init_eval(self):
        device = self.device
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32, device=device)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32, device=device)
        self.eval_target_return = None
        self.eval_timesteps = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.eval_episode_return, self.eval_episode_length = 0, 0


    def save_net(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        file_path = os.path.join(self.save_path, "dt.pt")
        torch.save(self.state_dict(), file_path)


    def save_jit(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        jit_model = torch.jit.script(self.to('cpu'))
        torch.jit.save(jit_model, os.path.join(self.save_path, 'dt_model.pth'))
        self.to(self.device)


    def load_net(self, device='cpu', model_path=''):
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)
        logger.info(f"Model loaded from {model_path}.")



    def set_hf32(self):
        if "npu" in self.device:
            import torch_npu
            torch_npu.npu.aclnn.allow_hf32 = self.hf32
            torch_npu.npu.conv.allow_hf32 = self.hf32
            torch_npu.npu.matmul.allow_hf32 = self.hf32

        elif "cuda" in self.device:
            torch.backends.cuda.matmul.allow_tf32 = self.hf32
            torch.backends.cudnn.allow_tf32 = self.hf32

        elif "mlu" in self.device:
            torch.backends.mlu.matmul.allow_tf32 = self.hf32
            torch.backends.cnnl.allow_tf32 = self.hf32
        logger.info(f"*************hf32: {self.hf32}*******************")
        logger.info(f"*********************compile:{self.compile}**********************")
        logger.info(f"*********************graph:{self.graph}**********************")

    def set_compile_model(self):
        if self.compile:
            if self.graph and self.shape_handle and "npu" in self.device:
                self.shape_options["triton.cudagraphs"] = True
                self.forward = torch.compile(
                    self.forward, backend="inductor", dynamic=self.enable_dynamic_compile,
                    options=self.shape_options
                )
            elif self.shape_handle and "npu" in self.device:
                self.forward = torch.compile(
                    self.forward, backend="inductor", dynamic=self.enable_dynamic_compile,
                    options=self.shape_options
                )
            elif self.graph:
                self.forward = torch.compile(
                    self.forward, backend="inductor", dynamic=self.enable_dynamic_compile,
                    mode="reduce-overhead"
                )
            else:
                self.forward = torch.compile(
                    self.forward, backend="inductor", dynamic=self.enable_dynamic_compile
                )
        else:
            self.forward = self.forward

    def is_manual_graph(self):
        return not self.compile and self.graph and ("npu" in self.device or "cuda" in self.device)

    def synchronize(self):
        if "npu" in self.device:
            torch.npu.synchronize()
        elif "cuda" in self.device:
            torch.cuda.synchronize()
        elif "mlu" in self.device:
            torch.mlu.synchronize()

    def model_infer_graph(self, inputs, batch_size):
        if isinstance(inputs, DTInputs):
            dt_inputs = inputs
        elif isinstance(inputs, dict):
            dt_inputs = DTInputs(
                states=inputs["states"],
                actions=inputs["actions"],
                rewards=inputs["rewards"],
                returns_to_go=inputs["returns_to_go"],
                timesteps=inputs["timesteps"],
                attention_mask=inputs.get("attention_mask", None),
            )
        else:
            raise TypeError(f"inputs must be DTInputs or dict, got {type(inputs)}")
            
        if batch_size not in self.graphs:
            if "npu" in self.device.lower():
                graph = torch.npu.NPUGraph()
                stream = torch.npu.Stream(self.device)
                stream_ctx = torch.npu.stream
            else:
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream(self.device)
                stream_ctx = torch.cuda.stream
            
            def _clone_contig(x: torch.Tensor) -> torch.Tensor:
                return x.clone().to(self.device, non_blocking=True).contiguous()

            static_dt_inputs = DTInputs(
                states=_clone_contig(dt_inputs.states),
                actions=_clone_contig(dt_inputs.actions),
                rewards=_clone_contig(dt_inputs.rewards),
                returns_to_go=_clone_contig(dt_inputs.returns_to_go),
                timesteps=_clone_contig(dt_inputs.timesteps),
                attention_mask=_clone_contig(dt_inputs.attention_mask),
            )

            with stream_ctx(stream):
                self.eval()
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.forward(static_dt_inputs)
                self.synchronize()

                if "npu" in self.device.lower():
                    with torch.npu.graph(graph):
                        static_output = self.forward(static_dt_inputs)
                else:
                    with torch.cuda.graph(graph):
                        static_output = self.forward(static_dt_inputs)

            self.graphs[batch_size] = {
                "graph": graph,
                "stream": stream,
                "static_inputs": static_dt_inputs, 
                "static_output": static_output,
            }

        else:
            graph_data = self.graphs[batch_size]
            static_dt_inputs = graph_data["static_inputs"]
            stream = graph_data["stream"]

            with (torch.cuda.stream(stream) if "cuda" in self.device.lower() else torch.npu.stream(stream)):
                static_dt_inputs.states.copy_(
                    dt_inputs.states.to(self.device, non_blocking=True), non_blocking=True
                    )
                static_dt_inputs.actions.copy_(
                    dt_inputs.actions.to(self.device, non_blocking=True), non_blocking=True
                    )
                static_dt_inputs.rewards.copy_(
                    dt_inputs.rewards.to(self.device, non_blocking=True), non_blocking=True
                    )
                static_dt_inputs.returns_to_go.copy_(
                    dt_inputs.returns_to_go.to(self.device, non_blocking=True), non_blocking=True
                    )
                static_dt_inputs.timesteps.copy_(
                    dt_inputs.timesteps.to(self.device, non_blocking=True), non_blocking=True
                    )
                static_dt_inputs.attention_mask.copy_(
                    dt_inputs.attention_mask.to(self.device, non_blocking=True), non_blocking=True
                    )

        graph_data = self.graphs[batch_size]
        with (torch.cuda.stream(graph_data["stream"]) 
            if "cuda" in self.device.lower() 
            else torch.npu.stream(graph_data["stream"])):
            graph_data["graph"].replay()

        self.synchronize()
        return graph_data["static_output"]

import os
import sys

import gymnasium as gym

from typing import Any, Dict, Optional
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
from stable_baselines3.dqn.dqn import DQN

from sumo_rl import SumoEnvironment

class PrioritizedReplayBuffer:    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def __len__(self):
        return self.size
    
    def add(self, state, action, reward, next_state, done, error=None):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        if self.size == 0:
            return None
            
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        total = self.size
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, errors, epsilon=1e-5):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + epsilon


class MyDQN(DQN):
    def __init__(
        self,
        *args,
        buffer_size: int = 50000,
        learning_starts: int = 1000,
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_beta: float = 0.4,
        prioritized_replay_beta_increment: float = 0.001,
        **kwargs
    ):
        super().__init__(*args, buffer_size=buffer_size, learning_starts=learning_starts, **kwargs)
                
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=prioritized_replay_alpha,
            beta=prioritized_replay_beta,
            beta_increment=prioritized_replay_beta_increment
        )
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.train()
        
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            if replay_data is None:
                continue
                
            states, actions, rewards, next_states, dones, indices, weights = replay_data
            
            states_tensor = torch.as_tensor(states, device=self.device)
            actions_tensor = torch.as_tensor(actions, device=self.device).unsqueeze(1)
            rewards_tensor = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
            next_states_tensor = torch.as_tensor(next_states, device=self.device)
            dones_tensor = torch.as_tensor(dones, device=self.device).unsqueeze(1)
            weights_tensor = torch.as_tensor(weights, device=self.device).unsqueeze(1)
            
            with torch.no_grad():
                next_actions = self.q_net(next_states_tensor).max(1)[1].unsqueeze(1)
                next_q_values = self.q_net_target(next_states_tensor).gather(1, next_actions)

                target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
            
            current_q_values = self.q_net(states_tensor).gather(1, actions_tensor)
            
            td_errors = (target_q_values - current_q_values).detach().cpu().numpy().flatten()
            
            self.replay_buffer.update_priorities(indices, td_errors)
            
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            loss = (loss * weights_tensor).mean()
            
            self.policy.optimizer.zero_grad()
            loss.backward()
            
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.policy.optimizer.step()
            losses.append(loss.item())
            
            self._on_step()
 
    def _on_step(self) -> None:
        super()._on_step()
    
    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 4,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MyDQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

env = SumoEnvironment(
    net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
    single_agent=True,
    route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
    out_csv_name="outputs/big-intersection/dqn_hw1",
    use_gui=False,
    num_seconds=5000,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    learning_starts=0,
    buffer_size=50000,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
)
model.learn(total_timesteps=100000)

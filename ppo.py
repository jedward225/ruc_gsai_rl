import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from env import MahjongGBEnv
from feature import FeatureAgent
from model import ResnetModel
from torch import nn
from collections import defaultdict
import random
import os
import logging
import matplotlib.pyplot as plt
import warnings

# 计算每个时间步的回报和优势
def compute_returns_and_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    returns = []
    advantages = []
    G = 0
    A = 0

    for i in reversed(range(len(rewards))):
        if dones[i]:
            G = 0
            A = 0

        G = rewards[i] + gamma * G
        returns.insert(0, G)

        if i < len(values) - 1:
            next_value = values[i + 1]
        else:
            next_value = 0

        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        A = delta + gamma * lam * A * (1 - dones[i])
        advantages.insert(0, A)

    return returns, advantages


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    returns = []
    gae = 0
    next_value = 0

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
        returns.insert(0, gae + values[i])

    return returns, advantages


# PPO 更新过程
def ppo_update(policy_network, optimizer, old_log_probs, states, actions, returns, advantages, action_masks,
               clip_param):
    returns = returns.detach()
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()

    dataset = TensorDataset(states, actions, returns, old_log_probs, advantages, action_masks)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(4):
        for batch in dataloader:
            state_batch, action_batch, return_batch, old_log_prob_batch, advantage_batch, action_mask_batch = batch

            input_dict = {'is_training': True, 'obs': {'observation': state_batch, 'action_mask': action_mask_batch}}
            action_logits, value = policy_network(input_dict)
            action_prob = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action_batch)

            # 1. 计算 ratio：新策略概率 / 旧策略概率
            ratio = torch.exp(log_prob - old_log_prob_batch)

            # 2. 计算裁剪后的 ratio
            clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)

            # 3. 计算 policy_loss：取裁剪前后的最小值，再取负（因为要最大化目标）
            policy_loss = -torch.min(
                ratio * advantage_batch,
                clipped_ratio * advantage_batch
            ).mean()

            # 4. 计算 value_loss：预测值与实际回报的均方误差
            value_loss = ((value.squeeze() - return_batch) ** 2).mean()

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()  # 确保没有重复调用 backward
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)


def reinforce_update(policy_network, optimizer, old_log_probs, states, actions, returns, advantages, action_masks,
               clip_param):
    returns = returns.detach()
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()

    dataset = TensorDataset(states, actions, returns, old_log_probs, advantages, action_masks)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(4):
        for batch in dataloader:
            state_batch, action_batch, return_batch, old_log_prob_batch, advantage_batch, action_mask_batch = batch

            input_dict = {'is_training': True, 'obs': {'observation': state_batch, 'action_mask': action_mask_batch}}
            action_logits, value = policy_network(input_dict)
            action_prob = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action_batch)

            policy_loss = -(log_prob * return_batch).mean()
            value_loss = policy_loss

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()  # 确保没有重复调用 backward
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)

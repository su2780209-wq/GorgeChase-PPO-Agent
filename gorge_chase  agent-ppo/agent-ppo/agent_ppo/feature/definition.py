#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions, GAE computation for Gorge Chase PPO.
峡谷追猎 PPO 数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls, attached
from agent_ppo.conf.conf import Config


# ObsData: feature=40D vector, legal_action=8D mask / 特征向量与合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: action, d_action(greedy), prob, value, attention_weights / 动作、贪心动作、概率、价值、注意力权重
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None, attention_weights=None)

# SampleData: single-frame sample with int dims / 单帧样本（整数表示维度）
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,  # 2D: [survival_reward, collection_reward]
    reward_sum=Config.VALUE_NUM,  # 2D
    done=1,
    value=Config.VALUE_NUM,  # 2D: [survival_value, collection_value]
    next_value=Config.VALUE_NUM,  # 2D
    advantage=Config.VALUE_NUM,  # 2D
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation) for multi-value heads.

    计算广义优势估计（GAE），支持多价值头。
    """
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    
    # Initialize GAE for each value head / 为每个价值头初始化GAE
    gae_survival = 0.0
    gae_collection = 0.0
    
    for sample in reversed(list_sample_data):
        # Ensure arrays are accessed safely / 安全访问数组
        val = np.array(sample.value, dtype=np.float32).flatten()
        rew = np.array(sample.reward, dtype=np.float32).flatten()
        nxt = np.array(sample.next_value, dtype=np.float32).flatten()
        
        # Pad to 2 if necessary (for backward compatibility) / 必要时填充到2维
        if len(val) < 2: val = np.pad(val, (0, 2 - len(val)))
        if len(rew) < 2: rew = np.pad(rew, (0, 2 - len(rew)))
        if len(nxt) < 2: nxt = np.pad(nxt, (0, 2 - len(nxt)))
        
        # Survival value GAE / 生存价值GAE
        delta_survival = -val[0] + rew[0] + gamma * nxt[0]
        gae_survival = gae_survival * gamma * lamda + delta_survival
        
        # Collection value GAE / 收集价值GAE
        delta_collection = -val[1] + rew[1] + gamma * nxt[1]
        gae_collection = gae_collection * gamma * lamda + delta_collection
        
        # Update sample attributes / 更新样本属性
        sample.advantage = np.array([gae_survival, gae_collection], dtype=np.float32)
        sample.reward_sum = np.array([gae_survival + val[0], gae_collection + val[1]], dtype=np.float32)

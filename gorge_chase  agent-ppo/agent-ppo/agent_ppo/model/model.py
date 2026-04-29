#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn
import numpy as np

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class AttentionMask(nn.Module):
    """Attention mechanism with feature grouping and masking.
    
    基于特征分组的注意力掩码机制。
    Dynamically weights different feature groups based on their importance.
    根据重要性动态加权不同的特征组。
    """
    
    def __init__(self, feature_dims, hidden_dim):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_groups = len(feature_dims)
        self.hidden_dim = hidden_dim
        
        # Project each feature group to common dimension / 将每个特征组投影到共同维度
        self.group_projections = nn.ModuleList([
            make_fc_layer(dim, hidden_dim // 2) for dim in feature_dims
        ])
        
        # Attention score computation / 注意力分数计算
        self.attention_net = nn.Sequential(
            make_fc_layer(hidden_dim // 2, 32),
            nn.ReLU(),
            make_fc_layer(32, 1)
        )
        
        # Temperature for softmax / softmax温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
    
    def forward(self, obs):
        """Compute attention-weighted features.
        
        计算注意力加权的特征。
        Args:
            obs: Input observation tensor [batch_size, total_features]
        Returns:
            weighted_features: Attention-weighted feature representation
            attention_weights: Attention weights for each group (for monitoring)
        """
        batch_size = obs.shape[0]
        
        # Split input into feature groups / 将输入分割为特征组
        feature_groups = []
        start_idx = 0
        for dim in self.feature_dims:
            end_idx = start_idx + dim
            group_feat = obs[:, start_idx:end_idx]  # [batch, group_dim]
            feature_groups.append(group_feat)
            start_idx = end_idx
        
        # Project each group to common space / 将每个组投影到共同空间
        projected_groups = []
        for i, (group_feat, projection) in enumerate(zip(feature_groups, self.group_projections)):
            projected = projection(group_feat)  # [batch, hidden_dim//2]
            projected = torch.relu(projected)
            projected_groups.append(projected)
        
        # Compute attention scores / 计算注意力分数
        attention_scores = []
        for projected in projected_groups:
            score = self.attention_net(projected).squeeze(-1)  # [batch]
            attention_scores.append(score)
        
        # Stack and apply temperature-scaled softmax / 堆叠并应用温度缩放的softmax
        attention_scores = torch.stack(attention_scores, dim=1)  # [batch, num_groups]
        attention_weights = torch.softmax(attention_scores / self.temperature, dim=1)
        
        # Weight and combine projected features / 加权组合投影后的特征
        weighted_features = torch.zeros(batch_size, self.hidden_dim // 2, device=obs.device)
        for i, projected in enumerate(projected_groups):
            weight = attention_weights[:, i:i+1]  # [batch, 1]
            weighted_features += weight * projected
        
        return weighted_features, attention_weights


class Model(nn.Module):
    """Enhanced model with attention masking mechanism.
    
    带注意力掩码机制的增强模型。
    Uses attention to dynamically focus on important features (monsters, treasures, etc.).
    使用注意力机制动态关注重要特征(怪物、宝箱等)。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_attention"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION  # 67D
        hidden_dim = 256
        mid_dim = 128
        action_num = Config.ACTION_NUM  # 16
        value_num = Config.VALUE_NUM
        
        # Feature group dimensions for attention / 注意力特征组维度
        # Must match Config.FEATURES order / 必须与Config.FEATURES顺序一致
        self.feature_groups = Config.FEATURES.copy()

        # Attention masking layer / 注意力掩码层
        self.attention_mask = AttentionMask(self.feature_groups, hidden_dim)

        # Enhanced shared backbone with residual connection / 带残差连接的增强骨干网络
        self.backbone = nn.Sequential(
            make_fc_layer(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.LayerNorm(mid_dim),
        )
        
        # Residual projection if needed / 残差投影(如需要)
        self.residual_proj = make_fc_layer(hidden_dim // 2, mid_dim)

        # Actor head / 策略头
        self.actor_head = make_fc_layer(mid_dim, action_num)

        # Dual Critic heads / 双价值头
        self.survival_critic_head = make_fc_layer(mid_dim, 1)  # Survival value / 生存价值
        self.collection_critic_head = make_fc_layer(mid_dim, 1)  # Collection value / 收集价值

    def forward(self, obs, inference=False):
        """Forward pass with attention masking and dual value heads.
        
        带注意力掩码和双价值头的前向传播。
        Args:
            obs: Observation tensor [batch_size, 67]
            inference: Whether in inference mode (returns attention weights)
        Returns:
            logits: Action logits
            value: State value [batch_size, 2] (survival + collection)
            attention_weights: (optional) Attention weights for monitoring
        """
        # Apply attention masking / 应用注意力掩码
        attended_features, attention_weights = self.attention_mask(obs)
        
        # Process through backbone / 通过骨干网络处理
        hidden = self.backbone(attended_features)
        
        # Add residual connection / 添加残差连接
        residual = self.residual_proj(attended_features)
        hidden = hidden + residual
        
        # Actor head / 策略头
        logits = self.actor_head(hidden)
        
        # Dual Critic heads / 双价值头
        survival_value = self.survival_critic_head(hidden)  # [batch, 1]
        collection_value = self.collection_critic_head(hidden)  # [batch, 1]
        value = torch.cat([survival_value, collection_value], dim=1)  # [batch, 2]
        
        if inference:
            return logits, value, attention_weights
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度（共 165 维）
    FEATURES = [
        4,   # hero_self: 位置、闪现冷却、buff剩余
        5,   # monster_1: 是否在视野、位置、速度、距离
        5,   # monster_2: 同上
        10,  # treasures: 最近3个宝箱信息(方向、距离、是否可见)×3 + 总数量
        121, # map_local: 11x11局部地图通行性
        8,   # path_connectivity: 8个方向的前方道路连通性评分（防死胡同）
        16,  # legal_action: 16维合法动作掩码
        2,   # progress: 步数归一化、存活比例
        2,   # monster_direction: 两个怪物的相对方向编码
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8个移动方向 + 8个闪现方向
    ACTION_NUM = 16

    # Value head / 价值头：双头（生存价值 + 收集价值）
    VALUE_NUM = 2

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.005  # Increased entropy coefficient for larger action space
    CLIP_PARAM = 0.2
    VF_COEF = 0.5  # Reduced value function coefficient to balance policy learning
    VF_COEF_SURVIVAL = 0.6  # Increased survival value weight / 提高生存价值权重
    VF_COEF_COLLECTION = 0.4  # Increased collection value weight / 提高收集价值权重（原0.3）
    GRAD_CLIP_RANGE = 0.5

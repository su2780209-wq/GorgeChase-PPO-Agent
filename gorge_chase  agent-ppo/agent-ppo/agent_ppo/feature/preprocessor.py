#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 增强版特征预处理与奖励设计。
"""

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Local map size / 局部地图大小(11x11)
LOCAL_MAP_SIZE = 11
# Number of treasure features / 宝箱特征数量(最近3个)
NUM_TREASURE_FEATURES = 3

# Directions for obstacle distance calculation (clockwise starting from Right)
# 方向定义：右, 右上, 上, 左上, 左, 左下, 下, 右下
# (row_offset, col_offset)
DIRECTIONS = [
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
    (0, -1), (1, -1), (1, 0), (1, 1)
]


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist_norm = 0.5
        self.visited_positions = set()  # Track visited positions for exploration reward
        self.last_treasure_count = 0  # Track treasure collection

    def _encode_direction(self, rel_dir):
        """Encode relative direction (0-8) to one-hot vector (8D).
        
        将相对方向(0-8)编码为one-hot向量(8维)。
        0=重叠/无效, 1=东, 2=东北, 3=北, 4=西北, 5=西, 6=西南, 7=南, 8=东南
        """
        encoding = np.zeros(8, dtype=np.float32)
        if 1 <= rel_dir <= 8:
            # Map direction index: 1->0(E), 2->1(NE), 3->2(N), 4->3(NW), 5->4(W), 6->5(SW), 7->6(S), 8->7(SE)
            encoding[rel_dir - 1] = 1.0
        return encoding

    def _calculate_path_connectivity(self, map_info, center_row, center_col):
        """Calculate path connectivity score in 8 directions (avoid dead-ends).
        
        计算8个方向的前方道路连通性评分（识别死胡同）。
        Returns: 8D array of connectivity scores [0-1].
        """
        if map_info is None:
            return np.zeros(8, dtype=np.float32)
        
        rows = len(map_info)
        cols = len(map_info[0])
        scores = np.zeros(8, dtype=np.float32)
        
        # Thresholds for scoring / 评分阈值
        # Score = min(path_length / 5.0, 1.0) -> 5格以上视为通畅
        max_eval_dist = 5.0
        
        for i, (dr, dc) in enumerate(DIRECTIONS):
            r, c = center_row + dr, center_col + dc
            path_len = 0.0
            # Raycast forward / 向前射线检测
            while 0 <= r < rows and 0 <= c < cols:
                if map_info[r][c] == 0:  # Hit obstacle / 遇到障碍
                    break
                path_len += 1.0
                r += dr
                c += dc
            
            # Calculate score / 计算评分
            # Short path (< 2) = dead end risk (score ~0.4)
            # Medium path (3-4) = moderate (score ~0.6-0.8)
            # Long path (>= 5) = open road (score 1.0)
            scores[i] = min(path_len / max_eval_dist, 1.0)
        
        return scores

    def _get_nearest_treasures(self, organs, hero_pos, max_count=3):
        """Extract features of nearest treasures.
        
        提取最近宝箱的特征。
        Returns: list of [direction_encoded(8D), distance_norm(1D), is_visible(1D)] for each treasure
        """
        treasures = [org for org in organs if org.get("sub_type") == 1 and org.get("status") == 1]
        
        if not treasures:
            return [], 0
        
        # Calculate distances and sort / 计算距离并排序
        for t in treasures:
            t_pos = t["pos"]
            t["_dist"] = np.sqrt((hero_pos["x"] - t_pos["x"])**2 + (hero_pos["z"] - t_pos["z"])**2)
        
        treasures.sort(key=lambda x: x["_dist"])
        nearest = treasures[:max_count]
        
        return nearest, len(treasures)

    def feature_process(self, env_obs, last_action):
        """Process env_obs into enhanced feature vector, legal_action mask, and reward.

        将 env_obs 转换为增强特征向量、合法动作掩码和即时奖励。
        """
        # Store last_action for flash usage detection / 保存 last_action 用于检测闪现使用
        self.last_action = last_action
        
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        monster_dirs = []
        min_dist_norm = 1.0
        
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                rel_dir = m.get("hero_relative_direction", 0)
                
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                    min_dist_norm = min(min_dist_norm, dist_norm)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
                monster_dirs.append(rel_dir)
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))
                monster_dirs.append(0)

        # Treasure features (10D) / 宝箱特征
        organs = frame_state.get("organs", [])
        nearest_treasures, total_treasures = self._get_nearest_treasures(organs, hero_pos, NUM_TREASURE_FEATURES)
        
        treasure_feat = np.zeros(NUM_TREASURE_FEATURES * 3 + 1, dtype=np.float32)  # 3*(dir+dist+vis) + count
        for idx, t in enumerate(nearest_treasures):
            if idx >= NUM_TREASURE_FEATURES:
                break
            base_idx = idx * 3
            t_pos = t["pos"]
            
            # Direction encoding (use first 2 dims of 8D direction as simplified representation)
            rel_dir = t.get("hero_relative_direction", 0)
            dir_encoding = self._encode_direction(rel_dir)
            treasure_feat[base_idx:base_idx+2] = dir_encoding[:2]  # Simplified to 2D
            
            # Distance normalization
            dist_norm = _norm(t["_dist"], MAP_SIZE * 1.41)
            treasure_feat[base_idx + 2] = dist_norm
        
        # Total visible treasure count (normalized)
        treasure_feat[-1] = _norm(total_treasures, 10.0)

        # Local map features (121D = 11x11) / 局部地图特征
        map_feat = np.zeros(LOCAL_MAP_SIZE * LOCAL_MAP_SIZE, dtype=np.float32)
        center_row, center_col = 0, 0
        if map_info is not None and len(map_info) >= LOCAL_MAP_SIZE:
            center_row = len(map_info) // 2
            center_col = len(map_info[0]) // 2
            half_size = LOCAL_MAP_SIZE // 2
            flat_idx = 0
            for row in range(center_row - half_size, center_row + half_size + 1):
                for col in range(center_col - half_size, center_col + half_size + 1):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1
        
        # Path connectivity features (8D) / 路径连通性特征（防死胡同）
        path_connectivity_feat = self._calculate_path_connectivity(
            map_info, center_row, center_col
        )

        # Legal action mask (16D) / 合法动作掩码
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Monster direction encoding (2D) / 怪物方向编码
        monster_dir_feat = np.array([
            _norm(monster_dirs[0], 8.0),  # Normalize direction 0-8 to 0-1
            _norm(monster_dirs[1], 8.0)
        ], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,           # 4D
                monster_feats[0],    # 5D
                monster_feats[1],    # 5D
                treasure_feat,       # 10D
                map_feat,            # 121D (11x11)
                path_connectivity_feat, # 8D (Path connectivity scores)
                np.array(legal_action, dtype=np.float32),  # 16D
                progress_feat,       # 2D
                monster_dir_feat,    # 2D
            ]
        )

        # Enhanced reward calculation / 增强奖励计算
        cur_min_dist_norm = min_dist_norm
        dist_delta = cur_min_dist_norm - self.last_min_monster_dist_norm
        
        # 1. Base survival reward / 基础生存奖励
        survive_reward = 0.02
        
        # 2. Distance shaping reward (ENHANCED) / 距离塑形奖励（增强）
        # Increased weight and added asymmetric penalty for moving closer / 增加权重并添加靠近怪物的非对称惩罚
        if dist_delta > 0:
            # Moving away from monster: positive reward / 远离怪物：正奖励
            dist_shaping = 0.3 * dist_delta
        else:
            # Moving toward monster: stronger penalty / 靠近怪物：更强惩罚
            dist_shaping = 0.5 * dist_delta  # Negative value
        
        # 3. Treasure collection reward / 宝箱收集奖励
        current_treasure_count = hero.get("treasure_collected_count", 0)
        treasure_reward = 0.0
        if current_treasure_count > self.last_treasure_count:
            treasure_reward = 1.5 * (current_treasure_count - self.last_treasure_count)  # Increased from 1.0
            self.last_treasure_count = current_treasure_count
        
        # 4. Exploration reward / 探索奖励
        current_pos = (int(hero_pos["x"]), int(hero_pos["z"]))
        explore_reward = 0.0
        if current_pos not in self.visited_positions:
            explore_reward = 0.08  # Increased from 0.05 to encourage exploration
            self.visited_positions.add(current_pos)
            # Limit memory size to prevent excessive growth
            if len(self.visited_positions) > 500:
                self.visited_positions.clear()
        
        # 5. Danger penalty (EARLIER WARNING) / 危险惩罚（提前预警）
        danger_penalty = 0.0
        if cur_min_dist_norm < 0.1:
            # Critical danger: very close to monster / 极度危险
            danger_penalty = -0.3
        elif cur_min_dist_norm < 0.2:
            # Warning zone: monster getting close / 警告区域
            danger_penalty = -0.1
        
        # 6. Flash encouragement reward (ENHANCED) / 闪现鼓励奖励（增强）
        flash_reward = 0.0
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        flash_available = (flash_cd_norm < 0.05)
        
        # 6.1 Encouragement when available / 可用时的基础鼓励
        if flash_available:
            if cur_min_dist_norm < 0.15:
                flash_reward = 0.15
            elif cur_min_dist_norm < 0.30:
                flash_reward = 0.05
        
        # 6.2 Flash usage success reward (NEW) / 闪现使用成功奖励（新增）
        # Check if last action was a flash (action 8-15) / 检查上一步是否为闪现动作
        if 8 <= self.last_action <= 15:
            # If distance increased significantly after flash, big reward / 闪现后距离显著拉开给予大额奖励
            dist_improvement = cur_min_dist_norm - self.last_min_monster_dist_norm
            if dist_improvement > 0.05:  # Successfully escaped / 成功逃脱
                flash_reward += 2.0  # Strong positive reinforcement / 强正反馈
            elif dist_improvement > 0:  # Slight improvement / 轻微改善
                flash_reward += 0.5
        
        # 7. Buff collection reward / buff收集奖励(新增)
        buff_reward = 0.0
        current_buff_time = hero.get("buff_remaining_time", 0)
        if not hasattr(self, 'last_buff_time'):
            self.last_buff_time = 0.0
        if current_buff_time > self.last_buff_time and current_buff_time > 40:
            buff_reward = 0.5
        self.last_buff_time = current_buff_time
        
        # 8. Milestone survival bonus / 里程碑生存奖励（适配新难度）
        # Reward for surviving critical phases / 存活关键阶段的奖励
        milestone_bonus = 0.0
        if self.step_no == 200:
            # Second monster appears (Advanced from 300) / 第二个怪物出现（提前至200）
            milestone_bonus = 0.5
        elif self.step_no == 350:
            # Pre-speedup warning (Advanced from 450) / 怪物加速前预警
            milestone_bonus = 0.3
        elif self.step_no == 400:
            # Monster speedup survived (Advanced from 500) / 成功度过怪物加速
            milestone_bonus = 0.8
        elif self.step_no == 600:
            # Mid-game survival buffer / 中期生存缓冲
            milestone_bonus = 0.5
        elif self.step_no == 800:
            # Late game survival / 后期生存
            milestone_bonus = 0.5
        elif self.step_no == 1000:
            # Full survival / 完全存活
            milestone_bonus = 1.0
        
        self.last_min_monster_dist_norm = cur_min_dist_norm

        # Multi-objective reward decomposition / 多目标奖励分解
        # Survival-related rewards / 生存相关奖励
        survival_reward = survive_reward + dist_shaping + danger_penalty + flash_reward + milestone_bonus
        
        # Collection-related rewards / 收集相关奖励
        collection_reward = treasure_reward + explore_reward + buff_reward
        
        # Return 2D reward vector / 返回2维奖励向量
        reward = [survival_reward, collection_reward]

        return feature, legal_action, reward

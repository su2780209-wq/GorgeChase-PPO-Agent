#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import os
import time

import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset env / 重置环境
            env_obs = self.env.reset(self.usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                
                # Extract attention weights for monitoring (if available) / 提取注意力权重用于监控
                attention_weights = None
                if hasattr(act_data, 'attention_weights') and act_data.attention_weights is not None:
                    attention_weights = act_data.attention_weights
                
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get("reward", [0.0, 0.0]), dtype=np.float32)
                total_reward += float(reward[0]) + float(reward[1])  # Sum both rewards

                # Terminal reward / 终局奖励
                final_reward = np.zeros(2, dtype=np.float32)  # 2D: [survival, collection]
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    step_no = env_obs["observation"].get("step_no", 0)

                    if terminated:
                        # Died: large penalty proportional to survival time
                        # 死亡惩罚：与存活时间成比例的大惩罚（仅影响生存价值）
                        survival_ratio = step_no / self.agent.preprocessor.max_step if self.agent.preprocessor.max_step > 0 else 0
                        final_reward[0] = -5.0 * (1.0 - survival_ratio)  # Survival penalty
                        final_reward[1] = 0.0  # No collection penalty
                        result_str = "FAIL"
                    else:
                        # Survived: bonus based on score and treasures
                        # 存活奖励：基于得分和宝箱数量的奖励
                        treasure_count = env_info.get("treasures_collected", 0)
                        final_reward[0] = 5.0  # Survival bonus
                        final_reward[1] = treasure_count * 0.5  # Collection bonus
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step_no} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f} treasures:{treasure_count if 'treasure_count' in locals() else 0}"
                    )

                # Build sample frame / 构造样本帧
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,  # 2D: [survival_reward, collection_reward]
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(2, dtype=np.float32),  # 2D
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:2],  # 2D: [survival_value, collection_value]
                    next_value=np.zeros(2, dtype=np.float32),  # 2D
                    advantage=np.zeros(2, dtype=np.float32),  # 2D
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                # Episode end / 对局结束
                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    # Monitor report / 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        treasure_count = env_info.get("treasures_collected", 0)
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "total_score": total_score,
                            "treasures_collected": treasure_count,
                        }
                        
                        # Add attention weights to monitoring if available / 如果有注意力权重则添加到监控
                        if attention_weights is not None:
                            feature_group_names = [
                                "hero_self", "monster_1", "monster_2", "treasures",
                                "map_local", "legal_action", "progress", "monster_dir"
                            ]
                            for i, (name, weight) in enumerate(zip(feature_group_names, attention_weights)):
                                monitor_data[f"attention_{name}"] = round(float(weight), 4)
                        
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info

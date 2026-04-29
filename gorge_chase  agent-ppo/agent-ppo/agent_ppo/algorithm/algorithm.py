#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
峡谷追猎 PPO 算法实现。

损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import os
import time

import torch
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM  # 2 (survival + collection)
        self.var_beta = Config.BETA_START
        self.clip_param = Config.CLIP_PARAM
        
        # Dual value head coefficients / 双价值头系数
        self.vf_coef_survival = Config.VF_COEF_SURVIVAL
        self.vf_coef_collection = Config.VF_COEF_COLLECTION

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """Training entry: PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行 PPO 更新。
        """
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        # Merge dual-head advantages into a single scalar for policy update / 合并双头优势为标量用于策略更新
        if len(advantage.shape) > 1 and advantage.shape[1] > 1:
            advantage = (advantage[:, 0] + advantage[:, 1]).view(-1, 1)
        elif len(advantage.shape) == 1:
            advantage = advantage.view(-1, 1)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss_survival": round(info_list[0].item(), 4),
                "value_loss_collection": round(info_list[1].item(), 4),
                "policy_loss": round(info_list[2].item(), 4),
                "entropy_loss": round(info_list[3].item(), 4),
                "reward_survival": round(reward[:, 0].mean().item(), 4),
                "reward_collection": round(reward[:, 1].mean().item(), 4),
                "train_step": self.train_step,
            }
            self.logger.info(
                f"[train] step:{self.train_step} total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss_survival:{results['value_loss_survival']} "
                f"value_loss_collection:{results['value_loss_collection']} "
                f"entropy:{results['entropy_loss']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 损失（策略损失 + 价值损失 + 熵正则化）。
        """
        # Masked softmax / 合法动作掩码 softmax
        prob_dist = self._masked_softmax(logits, legal_action)

        # Policy loss (PPO Clip) / 策略损失
        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage  # Already merged and shaped / 已合并并调整形状
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # Value loss (Clipped) for dual heads / 双价值头的裁剪价值损失
        vp = value_pred  # [batch, 2]
        ov = old_value  # [batch, 2]
        tdret = reward_sum  # [batch, 2]
        
        # Survival value loss / 生存价值损失
        vp_survival = vp[:, 0:1]
        ov_survival = ov[:, 0:1]
        tdret_survival = tdret[:, 0:1]
        value_clip_survival = ov_survival + (vp_survival - ov_survival).clamp(-self.clip_param, self.clip_param)
        value_loss_survival = (
            0.5
            * torch.maximum(
                torch.square(tdret_survival - vp_survival),
                torch.square(tdret_survival - value_clip_survival),
            ).mean()
        )
        
        # Collection value loss / 收集价值损失
        vp_collection = vp[:, 1:2]
        ov_collection = ov[:, 1:2]
        tdret_collection = tdret[:, 1:2]
        value_clip_collection = ov_collection + (vp_collection - ov_collection).clamp(-self.clip_param, self.clip_param)
        value_loss_collection = (
            0.5
            * torch.maximum(
                torch.square(tdret_collection - vp_collection),
                torch.square(tdret_collection - value_clip_collection),
            ).mean()
        )
        
        # Weighted total value loss / 加权总价值损失
        value_loss = self.vf_coef_survival * value_loss_survival + self.vf_coef_collection * value_loss_collection

        # Entropy loss / 熵损失
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        # Total loss / 总损失
        total_loss = value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, [value_loss_survival, value_loss_collection, policy_loss, entropy_loss]

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking (suppress illegal actions).

        合法动作掩码下的 softmax（将非法动作概率压为极小值）。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)

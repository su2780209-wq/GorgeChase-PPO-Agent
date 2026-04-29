# 🏃‍♂️ Gorge Chase PPO Agent (峡谷追猎强化学习智能体)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-PPO-brightgreen.svg)
![Framework](https://img.shields.io/badge/Framework-KaiwuDRL-orange.svg)

本项目是基于腾讯开悟（KaiwuDRL）分布式强化学习框架开发的**峡谷追猎（Gorge Chase）**赛题解决方案。项目采用 **Proximal Policy Optimization (PPO)** 算法，并在此基础上进行了深度的网络架构创新与特征工程优化，旨在控制智能体（鲁班七号）在复杂迷宫中躲避怪物追击并最大化收集宝箱。

## ✨ 核心技术创新 (Key Features)

### 1. 🧠 基于注意力掩码的神经网络 (Attention-Masked Network)
传统的 MLP 难以处理不同类型的异构特征（如自身状态、怪物方位、局部地图）。本项目在 `model.py` 中自定义了 `AttentionMask` 层：
* 将 165 维输入切分为 8 个逻辑特征组。
* 通过可学习的温度缩放 Softmax 动态计算各特征组的 Attention Weights。
* 主干网络引入了**残差连接 (Residual Connection)** 与 LayerNorm，大幅提升了深度网络的梯度传播稳定性。

### 2. 🎯 双价值头评估 (Dual-Head Critic)
生存（躲避怪物）和收集（吃宝箱）是两个时间尺度和方差截然不同的目标。本项目在 PPO 算法中实现了**双价值头机制**：
* **Survival Critic**: 评估生存价值，惩罚稠密但方差较小。
* **Collection Critic**: 评估收集价值，奖励稀疏且方差大。
* 在 `algorithm.py` 中对双头的 GAE (Generalized Advantage Estimation) 进行独立计算，最终按 `0.6` 和 `0.4` 的动态权重合并，有效避免了多目标强化学习中的梯度冲突。

### 3. 🗺️ 增强型空间特征提取 (Enhanced Spatial Features)
在 `preprocessor.py` 中，除了基础的归一化特征，引入了高度定制化的空间感知：
* **Path Connectivity (8方向连通性)**: 引入射线检测逻辑，计算 8 个方向的前方道路通畅度，**极大地降低了智能体走入死胡同的概率**。
* **11x11 Local Map**: 精细化局部通行性矩阵。
* **Treasure Attention**: 动态追踪最近的 3 个可见宝箱的位置与相对方向。

### 4. 🎁 极具诱导性的密集奖励重塑 (Dense Reward Shaping)
为了加速收敛并引导智能体学会高级微操（如极限闪现），设计了多维度的 Reward 函数：
* **Asymmetric Distance Penalty (非对称距离惩罚)**: 靠近怪物给予比远离怪物更重的惩罚。
* **Flash Encouragement (闪现正反馈)**: 鼓励在极度危险 (距离 < 0.15) 时使用闪现动作 (Action 8-15)，并在闪现成功拉开距离后给予 `+2.0` 的高额奖励。
* **Exploration Reward (探索奖励)**: 记录 `visited_positions`，鼓励智能体探索未知区域。
* **Milestone Bonus (里程碑奖励)**: 在游戏特定阶段（如怪物加速时、第二只怪物刷新时）存活，给予额外生存奖励。

## 📂 项目结构 (Repository Structure)

```text
├── agent_ppo/
│   ├── algorithm/algorithm.py      # 自定义 PPO 算法与双头 Loss 计算
│   ├── conf/train_env_conf.toml    # 环境参数 (地图、宝箱生成率、怪物加速节点)
│   ├── feature/preprocessor.py     # 特征提取、防死胡同逻辑与 Reward 函数
│   ├── model/model.py              # Actor-Critic 网络 (Attention + ResNet)
│   ├── workflow/train_workflow.py  # 训练循环与分布式数据推流
│   └── agent.py                    # 智能体主入口 (加载模型、执行推理)

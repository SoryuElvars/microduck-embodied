# Week 05：冻结 K10 Navigator 的观测侧 OOD 快筛

## 共同边界

本组包含四个**互不叠加**的高层观测因素。每个条件只改变 `RLNavigator` 接收的
`RobotState` 或 `GoalState`；MuJoCo 真值、冻结低层 ONNX、成功圈、路径、跌倒和终止判断均
不改变。因此 timeout 表示“策略按错误观测行动后，真值未能进入并保持在 0.20 m 成功圈”，
而不是把观测误差伪造成成功。

- 高层：冻结 K10 `model_1200.onnx`，SHA-256
  `554f420964daa7811ca09e1555d9711ffdf4df5a99318056db2cd987448c3642`
- 低层：冻结 `model_1500.onnx`，SHA-256
  `533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d`
- 每条件使用同一冻结的 10 个 Week04 EpisodeSpec；硬 Gate：成功率 100%、跌倒率 0、
  invalid-state rate 0。
- 所有协议均完成静态校验、6 个独立单测和 smoke。每个 episode 写入“输入/输出观测差”
  readback；dropout 还记录实际丢帧数。

## quick 结果

### 1. 固定 yaw 偏差

| Navigator yaw 偏差 | 成功 | 超时 | 跌倒 | Gate |
| --- | ---: | ---: | ---: | --- |
| 0° | 10/10 | 0 | 0 | 通过 |
| −5° | 10/10 | 0 | 0 | 通过 |
| +5° | 9/10 | 1 | 0 | **失败** |
| −10° | 10/10 | 0 | 0 | 通过 |
| +10° | 10/10 | 0 | 0 | 通过 |

结果不随偏差绝对值单调变化，说明 10 集 quick 只能揭示该场景集中的方向/场景组合脆弱性，
不能据此声称“10° 比 5° 更安全”。由于 `+5°` 已触发 timeout，本因素不升级 formal。

### 2. 机器人定位固定偏差

| RobotState world-frame 偏差 | 成功 | 超时 | 跌倒 |
| --- | ---: | ---: | ---: |
| 0 | 10/10 | 0 | 0 |
| x −5 cm / +5 cm | 7/10 / 5/10 | 3 / 5 | 0 / 0 |
| y −5 cm / +5 cm | 5/10 / 5/10 | 5 / 5 | 0 / 0 |
| x −10 cm / +10 cm | 7/10 / 3/10 | 3 / 7 | 0 / 0 |
| y −10 cm / +10 cm | 5/10 / 5/10 | 5 / 5 | 0 / 0 |

所有非零条件都未通过 Gate，且失败均为 timeout、无跌倒。这是位置误差会使策略在**估计**
目标附近减速/停住，而真值位置仍可能在成功圈外的直接证据。

### 3. GoalState 固定偏差

| GoalState world-frame 偏差 | 成功 | 超时 | 跌倒 |
| --- | ---: | ---: | ---: |
| 0 | 10/10 | 0 | 0 |
| x −5 cm / +5 cm | 5/10 / 7/10 | 5 / 3 | 0 / 0 |
| y −5 cm / +5 cm | 5/10 / 5/10 | 5 / 5 | 0 / 0 |
| x −10 cm / +10 cm | 3/10 / 7/10 | 7 / 3 | 0 / 0 |
| y −10 cm / +10 cm | 5/10 / 5/10 | 5 / 5 | 0 / 0 |

该矩阵与机器人定位偏差互为方向镜像：Navigator 使用相对目标向量 `goal_world - robot_world`，
所以“Goal +b”与“Robot −b”在高层输入上等价。所有非零条件均不通过 Gate，不升级 formal。

### 4. 高层 RobotState dropout

| 5 Hz 更新 dropout 概率 | 成功 | 超时 | 跌倒 | 实际丢帧数 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 0% | 10/10 | 0 | 0 | 0 | 通过 |
| 10% | 10/10 | 0 | 0 | 92 | 通过 |
| 20% | 10/10 | 0 | 0 | 152 | 通过 |

dropout 时只保持上一帧 `RobotState`，而目标和真值指标保持当前值。实际丢帧数确认 10%/20%
条件不是配置未生效。此因素的 quick 全部通过，若需要正式量化可由用户启动 full；本次未自动
启动 formal。

## 结论与下一步

固定位置/目标偏差是当前最明确的高层脆弱性，优先级高于 dropout。它们在相对位置上部分等价，
后续 domain-randomized 训练不能把两者同时当成两个独立因素混合；应先选择一个明确的状态估计
误差模型，例如每 episode 固定采样 RobotState `x/y` 偏置，并保留 GoalState 真值，才能归因。

本报告只提供 OOD 快筛证据，不构成 domain randomization 训练或 Sim2Real 成功声明。

## 原始证据

- `artifacts/week05/17_high_level_yaw_bias_ood_k10/quick/summary.json`，SHA-256
  `5af8509a6cbbec223c73538fe128d1867402b91de096f9a5dcb8d9b99bb96891`
- `artifacts/week05/18_high_level_robot_position_bias_ood_k10/quick/summary.json`，SHA-256
  `b5b7bf050a266bdd73b8e9de6ab84944327168e1c787e0c6e2205cdd56e036f9`
- `artifacts/week05/19_high_level_goal_position_bias_ood_k10/quick/summary.json`，SHA-256
  `5017c2c692bb6838c4245c61dc2f7da5f4f63df39b21795db988d8567c23f6f4`
- `artifacts/week05/20_high_level_observation_dropout_ood_k10/quick/summary.json`，SHA-256
  `48d877270d041b42a633e2c3692ae2c4ee4601bbc771ddd9f8be106a4a9d004d`

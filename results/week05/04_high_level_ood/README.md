# Week 05 / 04：旧低层上的高层 Navigator OOD

本报告只对应冻结 K10 `model_1200` 与旧低层 `model_1500`；扰动只作用于 Navigator 的命令交付或观测，不改变 MuJoCo 真值成功判定。

## 命令与决策延迟

高层命令通道 0/20/40 ms quick 各为 `10/10` 成功、零跌倒/超时；FIFO readback 确认实际延迟生效。以 200 ms 高层决策周期计，0/200/400/600 ms 各 `10/10`，800 ms 为 `9/10` 且 1 timeout。因此高层对数百毫秒的决策错位有一定耐受，但这些均是 quick 级边界，未与正式 Episode 合并。

## 观测侧 OOD

10%/20% RobotState dropout quick 均为 `10/10`，且实际丢帧数被记录。相反，RobotState 或 GoalState 的非零固定 x/y 偏差普遍造成 timeout；例如 ±5 cm 已出现 5--7/10 成功，±10 cm 最低 3/10。固定 yaw 偏差则呈非单调的场景敏感性，+5° 为 9/10。

结论是高层当前最明确的薄弱点是**固定定位/目标偏差**，不是短时 dropout。若后续做高层 DR，应只选择一个相对位置误差模型，按 episode 固定采样，并在新低层冻结后重新建立 Nominal baseline；不能把 Robot 与 Goal 的等价偏差同时叠加并声称独立收益。

原始证据位于：

- 命令 FIFO 延迟：`11_high_level_delay_ood_k10`、`12_high_level_delay_stress_ood_k10`、
  `13_high_level_delay_stress_100_120_k10`、`14_high_level_delay_stress_140_160_k10`；
- 决策延迟：`15_high_level_decision_delay_ood_k10`、`16_high_level_decision_delay_600_800_k10`；
- 观测扰动：`17_high_level_yaw_bias_ood_k10`、`18_high_level_robot_position_bias_ood_k10`、
  `19_high_level_goal_position_bias_ood_k10`、`20_high_level_observation_dropout_ood_k10`。

以上目录均位于 `artifacts/week05/`，全部是 smoke/quick 级证据，不与旧低层上的 200-Episode
checkpoint formal 合并。

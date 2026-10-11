# Week 05：RL PointGoal Navigator 与低层延迟 Domain Randomization

本周分为两条不可混淆的证据链：高层 RL Navigator 使用旧低层 `model_1500.onnx` 完成了训练、选模与 OOD 诊断；新的低层延迟 DR 候选 `model_1498.onnx` 随后完成了独立训练与 500-Episode OOD 回归。两者不能直接拼接成同一个已验证系统。

| 报告 | 内容 | 当前结论 |
|---|---|---|
| [01 Navigator 基础](01_navigator_foundation/README.md) | 7D→2D 接口、K 快筛与训练边界 | 分层训练链路已验证 |
| [02 停稳到达](02_stable_arrival/README.md) | 速度感知成功条件与 K5/K10 诊断 | 位置到达不等于停车 |
| [03 Reward 与选模](03_reward_and_model_selection/README.md) | 近目标减速奖励、课程学习与 K10 选模 | 旧低层上的历史高层冻结为 `model_1200` |
| [04 高层 OOD](04_high_level_ood/README.md) | 命令/决策延迟、观测误差与 dropout | 定位/目标固定偏差是主要缺口 |
| [05 低层延迟 DR 与 OOD](05_low_level_delay_dr_and_ood/README.md) | action-delay DR 训练、checkpoint sweep、500 Episode 回归 | `model_1498` 已冻结；先做旧高层接入诊断 |
| [06 新低层上的 Navigator 重训](06_new_low_level_navigator_retraining/README.md) | 64-env K5/K10 全课程 checkpoint quick、K2 Stage-1 诊断及 K5/K10 formal | K5 `model_1499` 为 175/200，K10 为 169/200；停止 K2 |

所有正式、quick 与 smoke 的 Episode 数分别记录在对应报告中，不能相加为同一个成功率。原始 JSON、逐步 CSV、训练 checkpoint 与 ONNX 均保留在 `artifacts/` 或官方训练日志中。

当前最重要的系统边界是：`model_1498` 冻结并替换旧低层后，必须在新低层上重新训练、选择和评测 Navigator；本周旧高层 `model_1200` 的 200-Episode 结论只适用于旧低层。

该边界现已由接入诊断验证：保持 `model_1200` 不变而仅替换为冻结 `model_1498` 后，配对 quick 从
10/10 变为 8/10，故不将该组合升级到 formal，而是转入新低层上的 Navigator 重训。

新低层上的 K5 与 K10 均已完成三阶段 1500-iteration 训练和同 checkpoint、同 EpisodeSpec 的
16-checkpoint quick。随后对两者的 `model_1499` 执行共享 200-Episode formal：K5 为
175/200，K10 为 169/200，两者均零跌倒、零 invalid、零到达后重新离开。配对结果中有 6 个
K5-only 成功、0 个 K10-only 成功（事后精确 McNemar 双侧 `p=0.03125`）。因此 K5 `model_1499`
是当前两个固定候选中的 nominal 首选，但只有 training seed 42，不能推广为 K5 架构普遍优于 K10。

## 原始证据目录覆盖

| 报告 | 对应 `artifacts/week05/` 目录 |
|---|---|
| 01 | `02_rl_quick` |
| 02 | `03_stable_arrival_rescore`、`04_stable_live`、`05_stable_k5_retrain_eval`、`06_stable_k10_retrain_eval` |
| 03 | `07_near-speed_k10_quick_eval`、`08_near-speed009_k10_quick_eval`、`09_near-speed008_k10_formal_eval`、`10_checkpoint_selection_k10` |
| 04 | `11_high_level_delay_ood_k10` 至 `16_high_level_decision_delay_600_800_k10`，以及 `17_high_level_yaw_bias_ood_k10` 至 `20_high_level_observation_dropout_ood_k10` |
| 05 | `14_low_level_action_delay_dr_quick` 至 `19_low_level_non_delay_ood_formal`、`20_frozen_low_level_high_level_integration`、`25_low_level_bridge_parity` |
| 06 | `21_low1498_k10_quick_training` 至 `24_low1498_k_screen_training`、`26_low1498_curr500_checkpoint_sweep`、`27_low1498_k2_k5_curr500_checkpoint_screen`、`28_low1498_k5_curr1500_checkpoint_sweep`、`29_low1498_k5_k10_formal` |

编号 14–20 同时存在高层与低层实验，必须按完整目录名区分。目录 21–24 的 1-env PPO 横向比较已在
报告 06 中标为混杂实验；保留原始证据，但不再用于 K 值结论。

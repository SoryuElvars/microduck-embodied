# Week 05 / 05：低层 Action-Delay Domain Randomization 与 500-Episode OOD 回归

## 训练与 checkpoint 选择

旧低层 `model_1500` 已有多因素随机化，但在部署侧 50 Hz 动作 FIFO 延迟上出现硬边界：60 ms 仅 8/25 到达、17/25 跌倒；80 ms 为 0/25 到达、25/25 跌倒。新分支只新增每 episode 固定采样 `0/20/40 ms` 的 action-delay FIFO，保留原 Reward、PPO、command distribution 与其他 DR。

从头训练的 `model_249` 在各延迟档都不跌倒但 0/25 到达，属于保守不动而非鲁棒。恢复训练 checkpoint sweep 显示 750 后开始形成可用行走，`model_1498` 成为唯一五档 quick 均 5/5 到达、零跌倒的候选。ONNX SHA-256：`44f61b6653a75506f52b68e890b75156e34ae68a8db9296102c575adeb558e65`。

## 完整 OOD 结果

本次将所有结果严格按条件独立 Gate 评估，共 500 Episode：

| 因素组 | 条件数 × Episode | 到达 / 跌倒 / 超时 / invalid | Gate |
|---|---:|---:|---|
| Nominal + action delay 0/20/40/60/80 ms | 5 × 25 = 125 | 125/125 / 0 / 0 / 0 | 全部通过 |
| 接触摩擦 0.3/0.5/0.7/0.9/1.1 | 5 × 25 = 125 | 125/125 / 0 / 0 / 0 | 全部通过 |
| 电机代理 80/90/110% + backlash 2° | 4 × 25 = 100 | 100/100 / 0 / 0 / 0 | 全部通过 |
| 质量/惯量 90/110%，IMU 与编码器噪声低/高档 | 6 × 25 = 150 | 150/150 / 0 / 0 / 0 | 全部通过 |
| **合计** | **500** | **500/500 / 0 / 0 / 0** | **全部通过** |

延迟 DR 的明确收益是 60/80 ms：新候选在 50 Episode formal 中全成功且无跌倒，旧 Control 在相同 quick 口径分别为 2/5 到达、3/5 跌倒和 0/5 到达、5/5 跌倒。

## 保留的质量边界

成功率不代表所有运动质量不变。新候选 Nominal `vx/wz` tracking RMSE 为 `0.155 m/s / 0.537 rad/s`，旧 Control 为 `0.123 / 0.486`，即存在小幅 Nominal tracking 退化。停稳残余量在 80 ms 为 `0.060 m/s / 0.283 rad/s`；高档编码器噪声为 `0.024 / 0.177`，backlash 为 `0.021 / 0.161`。它们未触发当前 Gate，但应作为接入新 Navigator 后的持续诊断项。

## 决策边界

`model_1498` 已冻结为低层候选；完整 ONNX 哈希、内嵌 normalizer、训练配置、上游 commit 和 500 Episode 证据见 [selected_low_level_model.json](selected_low_level_model.json)。高层训练还使用由同一 `model_1498.pt` 官方导出的 TorchScript，哈希也在清单中；它避免将 ONNX 与训练环境的 batched Torch policy 接口混用。冻结不改写旧高层的历史 `selected_model.json`，因为后者仍精确绑定旧低层 `model_1500`。

下一步先将冻结的 `model_1498` 接到不变的历史高层 `model_1200`，依次运行静态校验、smoke 与 10-Episode quick。这是兼容性诊断而非高层能力结论：只要低层命令响应改变，Navigator 仍须在新低层上重新训练、checkpoint sweep 和正式评测。

## 冻结低层接入历史高层：兼容性诊断

静态校验确认高层 `model_1200` 仍是 `[1, 7] -> [1, 2]`，新低层 `model_1498` 仍是
`[1, 61] -> [1, 14]`，两份 SHA 均与冻结清单匹配。5 秒 smoke 在两种低层下均会 timeout，
所以它只确认了加载、reset、命令链、终止和记录链正常；新低层的终点距目标 `1.113 m`，旧低层为
`0.951 m`，已显示速度响应较弱。

在完全相同的 10 个冻结 EpisodeSpec 上，旧组合是 `10/10` 到达、0 跌倒、0 超时；新低层与旧高层
组合为 `8/10` 到达、0 跌倒、2 超时、0 invalid。`w04e002` 与 `w04e009` 从原来的 success 变为
timeout；到达样本的中位完成时间从 `14.59 s` 增至 `17.70 s`，路径效率从 `0.696` 降至 `0.681`。
这不是安全性故障，却是明确的旧高层兼容性退化，故停止在 quick 级，不启动该组合的 formal。

原始结果：`artifacts/week05/20_frozen_low_level_high_level_integration/{smoke,quick}/summary.json`。
该低层现在是后续新 Navigator 训练唯一允许使用的冻结低层；历史 `model_1200` 仅保留为对照，不能
移植为新系统的候选。

## 新低层上的早期 Navigator 诊断与证据修正

固定指令旁路高层策略后，新低层在 Warp 训练环境中的 `vx=0.25 m/s` 实际均值分别为 K2
`0.1201`、K5 `0.1216`、K10 `0.1221 m/s`，三者均无跌倒；说明 K 不会改变低层对稳定命令的
执行能力，也没有发现归一化、符号或 rate-limit 桥接错误。相同 K10 条件下，旧低层实际前进速度
为 `0.1552 m/s`，新低层下降约 22%。这部分桥接证据仍有效，位于
`artifacts/week05/25_low_level_bridge_parity/`。

但是，最初 K2/K5/K10 的 100-iteration PPO 训练都使用了 `num_envs=1`。三个 K 每 iteration 的
样本数分别为 60、24、12，因此同时混入了不同 PPO batch size；早期 K10 10/10、K5 2/10、K2
0/10 不能再作为 K 值横向结论，原先关于短 K 探索困难的机制解释也降级为未验证假设。相关原始
训练诊断保留在 `artifacts/week05/21_low1498_k10_quick_training/` 至
`24_low1498_k_screen_training/`，但不用于选模。

后续已改用统一 64 environments、seed 42 和每阶段 500 iterations：K10 完成三阶段训练，K2/K5
完成 500-iteration Stage-1 对照。新的 checkpoint smoke/quick、K2 entropy 退化以及 K5 续训决策
统一记录在 [报告 06](../06_new_low_level_navigator_retraining/README.md)。

原始证据：延迟训练/checkpoint quick/formal 分别位于
`artifacts/week05/14_low_level_action_delay_dr_quick/`、
`15_low_level_action_delay_dr_checkpoint_quick/`、`16_low_level_action_delay_dr_formal/`；其余 OOD 的
smoke/quick/formal 分别位于 `17_low_level_non_delay_ood_smoke/`、
`18_low_level_non_delay_ood_quick/`、`19_low_level_non_delay_ood_formal/`。smoke/quick 不计入上述
500 Episode。

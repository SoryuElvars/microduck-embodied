# Week 05 / 14：低层 Action-Delay DR 的 250-iteration 快筛

## 结论

本次训练分支在原 `Yaw-Only` 低层配方上新增每 episode 固定、均匀采样的
`0 / 20 / 40 ms` 动作 FIFO 延迟；Reward、command distribution、PPO 与已有物理/传感器
Domain Randomization 保持不变。250 iteration 的最终 checkpoint `model_249.pt` 已通过
官方 normalizer-aware 导出为 ONNX，并与冻结的 `model_1500.onnx` 使用相同目标、seed、
Controller 和部署侧 delay 注入进行了配对 quick screen。

`model_249` 在五个条件下均没有跌倒或 invalid state，但总计 `0/25` 到达、`25/25`
timeout，Nominal 本身也是 `0/5`。这表示候选尚未学会基础任务；“不跌倒”是近似保守不动，
不能解释为延迟鲁棒性提高。快筛未通过任务升级条件，因此不运行 formal、不评测更早的
`model_125`，也不替换现有冻结低层。

## 冻结协议

- 评测配置：`evaluation/robustness_configs/week05_low_level_action_delay_dr_v1.json`
- 配置 SHA-256：`3207d5d764e77422100797501200b5a421f1899a16050c2d2be19f026de0d37e`
- 条件：Nominal、delay `20 / 40 / 60 / 80 ms`
- 每条件：五个对称目标 × seed 42，共 5 Episode
- `0/20/40 ms`：Nominal/训练支持回归；`60/80 ms`：未参与训练的压力 OOD
- 快筛安全 Gate：Fall Rate `=0`、invalid-state rate `=0`
- 到达率与 timeout 同时报告；安全 Gate 通过不等于任务能力通过
- 官方代码 commit：`062921c4c9f65107e391df042f8de424136213c3`

## 模型

| 模型 | 角色 | SHA-256 |
|---|---|---|
| `model_1500.onnx` | 冻结 Control | `533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d` |
| `model_249.pt` | 250-iteration 训练 checkpoint | `49607063ad6975f359a82bd732f4693865b5ac534ba4f02ada7acd38b22fce81` |
| `model_249.onnx` | 待评测候选 | `4094ffa1f94287040119b06b081aa4829163981e30df811ad7a85e88e9207ab5` |

训练 run：

```text
/home/elvars/projects/microduck_rl/logs/rsl_rl/
velocity_yaw_only_tracking_action_delay_dr/
2026-10-10_22-04-40_week05-action-delay-dr-quick-seed42
```

## 配对快筛结果

| 条件 | `model_1500` 到达/跌倒/超时 | `model_249` 到达/跌倒/超时 | `model_249` 中位最终距离 | 判读 |
|---|---:|---:|---:|---|
| Nominal | 5/0/0 | 0/0/5 | 1.031 m | 基础任务退化 |
| Delay 20 ms | 5/0/0 | 0/0/5 | 1.283 m | 训练支持内未学会 |
| Delay 40 ms | 5/0/0 | 0/0/5 | 1.098 m | 训练支持内未学会 |
| Delay 60 ms | 2/3/0 | 0/0/5 | 0.911 m | 无跌倒但未到达 |
| Delay 80 ms | 0/5/0 | 0/0/5 | 1.402 m | 无跌倒但未到达 |

Control 重现了既有边界：Nominal、20 ms、40 ms 均 `5/5` 到达且无跌倒；60 ms 为
`2/5` 到达、`3/5` 跌倒；80 ms 为 `0/5` 到达、`5/5` 跌倒。这证明本轮候选失败不是
评测协议或 Controller 失效造成。

`model_249` 的 Nominal 中位 tracking RMSE 为 `vx=0.205 m/s`、`wz=0.481 rad/s`，中位
最终距离仍为 `1.031 m`。它在更大延迟下保持站立不能作为进步，因为所有条件的任务成功率
均为零。

## 决策

1. 快筛失败，停止 formal；不将 quick Episode 与任何正式结果合并。
2. 保持 `model_1500.onnx` 为冻结 Control 与当前系统低层。
3. 250 iteration 对从头训练的低层步态明显不足。下一步应先检查 WandB 的速度跟踪主项、
   Episode length 与动作幅值，再决定继续从头训练到原 1500/2000-iteration 量级，还是改为
   从 `model_1500.pt` 做单因素延迟微调；不能仅凭 Return 上升直接续长训练。

原始逐 Episode CSV、进度与 suite summary 位于：

```text
artifacts/week05/14_low_level_action_delay_dr_quick/
```

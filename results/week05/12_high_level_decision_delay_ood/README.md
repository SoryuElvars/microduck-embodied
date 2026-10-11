# Week 05：冻结 K10 Navigator 的高层离散决策延迟快筛

## 注入语义与边界

这里测量的是**高层决策到实际执行**的延迟，而不是 Week 03 的低层 joint-action / actuator
delay。Navigator 在时刻 `t` 用当时的 `RobotState`、`GoalState` 产生命令；FIFO 让该命令在
`t + delay` 才送到冻结 locomotion policy。高层频率为 5 Hz，故决策周期是 200 ms，所有本
实验档位强制为 200 ms 的整数倍。

```text
RobotState + GoalState at t → RL Navigator → FIFO(0/200/... ms)
                                             ↓ at t + delay
                                  frozen locomotion ONNX → MuJoCo
```

- 高层策略：冻结 K10 `model_1200.onnx`，SHA-256
  `554f420964daa7811ca09e1555d9711ffdf4df5a99318056db2cd987448c3642`
- 低层策略：冻结 `model_1500.onnx`，SHA-256
  `533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d`
- 冻结基础场景：Week04 前 10 个 EpisodeSpec；成功、路径和安全指标仍由无扰动仿真真值计算。
- 每个 episode 的 delivery trace 都验证了实际 FIFO 步数：200/400/600/800 ms 分别等于
  10/20/30/40 个 20 ms 低层控制步。

此前的 `0/20/40 ms` 是**小于一个决策周期的命令传输延迟**；`60--160 ms` 的探索也属于这
一类，不应当作“少做了一拍高层决策”的证据。本报告只记录 200 ms 量化的决策延迟。

## 结果：smoke 与 quick

每个独立协议均先通过静态校验、单测和 5 秒 smoke；smoke 的短时 timeout 仅用于验证注入与
readback，未进入能力结论。quick 对每档运行相同的 10 个冻结 EpisodeSpec。硬 Gate 为
Success Rate 100%、Fall Rate 0、invalid-state rate 0；首次不通过即停止继续增加延迟。

| 离散决策延迟 | 对应高层周期 | 成功 | 跌倒 | 超时 | 中位完成时间 (s) | 中位路径效率 | 到达后重离开 | quick Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 ms | 0 | 10/10 | 0 | 0 | 14.59 | 0.696 | 0/10 | 通过 |
| 200 ms | 1 | 10/10 | 0 | 0 | 13.11 | 0.701 | 0/10 | 通过 |
| 400 ms | 2 | 10/10 | 0 | 0 | 12.95 | 0.722 | 0/10 | 通过 |
| 600 ms | 3 | 10/10 | 0 | 0 | 13.52 | 0.700 | 0/10 | 通过 |
| 800 ms | 4 | 9/10 | 0 | 1 | 13.60* | 0.694* | 0/9 | **失败** |

`*` 仅在到达成功的 episode 上计算。800 ms 的失败 episode 是 `w04e000` 的 timeout，最终距离
为 0.201 m；没有发生跌倒或 invalid state。

200--600 ms 偶然出现的完成时间/效率小幅变化不表示“延迟改善性能”：每档只有 10 集，且本
实验的目标是定位首次安全/任务 Gate 失败，而不是比较微小性能排序。

## 可得与不可得的结论

- 在这 10 个冻结 quick 场景上，离散高层决策到执行延迟在 **600 ms** 仍通过硬 Gate；
  **800 ms** 首次失败。因此快速筛查得到的边界是 `(600, 800] ms`，而不是精确的物理时延
  阈值，也不是实机安全保证。
- 这项结果不改变 Week03 的结论：低层 60 ms actuator delay 会失去平衡，与本实验的高层
  VelocityCommand 延迟是不同的注入位置和失效机制。
- 800 ms 已不满足 quick Gate，故不对该档或更高档启动 formal。原有 `0/20/40 ms` 命令传输
  延迟的 formal 仍是独立、由用户启动的 3 × 200 episode 评测，不能同此 quick 合并。

## 可复现证据

- [第一段协议](../../../evaluation/pointgoal_protocols/week05_rl_decision_delay_ood_v1.json)，
  SHA-256 `882cc72ce301e3db435c7aeef927455cfa65b5c46fc663fff39014231ccd0bef`
- [第二段协议](../../../evaluation/pointgoal_protocols/week05_rl_decision_delay_ood_v2_600_800.json)，
  SHA-256 `95c8f07696d7157391f05c6ed78a4171a197f10121376d05a8ef7de43eaa8c01`
- raw quick summaries（本地 `artifacts/`，不提交）：
  `artifacts/week05/15_high_level_decision_delay_ood_k10/quick/summary.json`，SHA-256
  `9c504e3bd2af0a505961fdbc74fbd8cb9821f92bafb9a3cc1261a0be6aed7f74`；
  `artifacts/week05/16_high_level_decision_delay_600_800_k10/quick/summary.json`，SHA-256
  `52a6f67d84f3d7b1675c5825170611a0a1e79a1f0923af92d048945e4c98bdd2`。

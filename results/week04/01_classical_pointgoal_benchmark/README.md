# Week 4 Classical PointGoal Benchmark v1：开发与快筛记录

## 当前结论

Classical PointGoal Benchmark v1 的代码、冻结场景清单、双 Controller 运行链路、
断点续跑和 post-arrival 指标已经建立。按照单变量原则，只把两个 Controller 共享的
`goal_tolerance` 从 `0.20 m` 改为 `0.15 m`；成功半径仍为 `0.20 m`，模型、场景、seeds、
增益、速度上限、timeout 和 Gate 均未改变。

2026-09-21 完成同一组 10 个随机场景 × 2 Controller quick screen。两者均为 10/10
到达、0 跌倒、0 超时、0 invalid state，且 post-arrival re-departure rate 从此前的
100% 降为 0%。这说明 `0.15 m` 停止阈值为制动和步态漂移提供了足够余量，解决了本轮
quick 暴露的重新出圈问题。

这仍是 quick 证据，不是正式 Controller 胜负结论。quick 通过后，两个 Controller
参数和协议已冻结；尚未运行 200 个共享场景 × 2 Controller 的 `--full`。

## 冻结输入与版本

| 项目 | 当前值 |
|---|---|
| Protocol | `evaluation/pointgoal_protocols/week04_classical_v1.json` |
| Protocol 状态 | `frozen` |
| Episode manifest 状态 | `frozen` |
| Episode manifest SHA-256 | `b2d4c62121e0c1b4f290094079512c09cf7f6eaf8e1918622f12483c2292337c` |
| Protocol 文件 SHA-256 | `69db75465d5958f6c41ff49691cec0ff849cc6c5521ae4c28264f2806aec5be7` |
| 底层策略 | `model_1500.onnx` |
| Policy SHA-256 | `533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d` |
| `microduck_rl` commit | `062921c4c9f65107e391df042f8de424136213c3` |
| Quick run signature（冻结前记录） | `ef0e2f22a14b0ed9a7538487bbf0bc2e587fdb0d70bafc56d02ccb98b5da531e` |
| Navigator action | `[vx, 0, wz]` |
| 共享速度上限 | `vx <= 0.25 m/s`，`abs(wz) <= 0.5 rad/s` |
| 共享停止阈值 | `goal_tolerance = 0.15 m` |
| 成功定义 | 进入 `0.20 m` 半径并连续保持 `0.5 s` |
| 到达后观察 | `2.0 s` |
| Reward | 不可用；这是 ONNX 部署形态评测 |

任务是位置型 PointGoal。`heading_error` 只表示朝向目标点的误差，不要求最终目标 yaw；
指定最终姿态属于独立的 PoseGoal 扩展。

冻结操作只改变协议状态和描述，因此 Episode manifest SHA 不变；Protocol 文件 SHA 随之
更新。上表 Quick run signature 保留冻结前 quick 的原始可追溯记录，正式评测会基于冻结
协议生成新的 run signature，旧 quick 进度不会被混入 `full`。

## 执行层级

### 静态校验

`--validate-only` 已通过，确认协议可加载、200 个 EpisodeSpec 的 reset seed 与 Start
一致、Controller tolerance 小于任务成功半径、模型文件存在、上游 commit 可记录，
预期 Actor I/O 为 `61 obs -> 14 actions`。

### Smoke

从冻结清单选择最接近正前、左、右、后方的 4 个场景，两个 Controller 各运行一次，
共 8 个 Controller-Episode。5 秒 smoke 截断下均为 timeout，但无 fall 或 invalid
state；它只验证 reset/goal 注入、双 Controller、轨迹、进度和汇总链路，不计入 quick
或 formal，也不用于能力结论。

### Quick screen

Quick 使用 `w04e000--w04e009` 和 reset seeds `40000--40009`。目标距离范围为
`1.019--1.791 m`，目标相对方向覆盖约 `-178.6°--+163.4°`，包含前方、侧方与后方
目标。两个 Controller 使用完全相同的 10 个 EpisodeSpec。

| 指标 | Naive P | Constrained |
|---|---:|---:|
| 共享场景数 | 10 | 10 |
| Success Rate | 100% | 100% |
| Fall / Timeout / Invalid | 0 / 0 / 0 | 0 / 0 / 0 |
| Median Final Distance | 0.1480 m | 0.1575 m |
| Median Path Length to Arrival | 2.2607 m | 1.8224 m |
| Median Path Efficiency | 0.6959 | 0.7313 |
| Median Completion Time | 14.43 s | 13.53 s |
| Mean `vx` Saturation Fraction | 0.7642 | 0.0000 |
| Mean `wz` Saturation Fraction | 0.2177 | 0.1951 |
| Median Post-arrival Displacement | 0.0212 m | 0.0146 m |
| Median Post-arrival Max Displacement | 0.0254 m | 0.0285 m |
| Post-arrival Re-departure Rate | 0% | 0% |
| Median Post-arrival Max Goal Distance | 0.1672 m | 0.1675 m |
| Maximum Post-arrival Goal Distance | 0.1698 m | 0.1717 m |

`Constrained - Naive P` 的逐场景配对中位差为：

| 配对差值 | 中位数 | 解释 |
|---|---:|---|
| Final Distance | +0.0096 m | Constrained 停得略远，但仍有充分成功余量 |
| Path Length to Arrival | -0.1188 m | Constrained 更短 |
| Path Efficiency | +0.0376 | Constrained 更高 |
| Completion Time | +0.20 s | 配对中位数下 Constrained 略慢 |
| `vx` Saturation Fraction | -0.7821 | Constrained 明显减少饱和 |
| `wz` Saturation Fraction | -0.0203 | Constrained 略少饱和 |
| Post-arrival Displacement | -0.0069 m | Constrained 从成功判定点到最终位置的位移更小 |

两组 Success 都是 10/10，因此 Success 胜负为 0/0，10 场均为平局。Quick 样本只用于
发现明显安全或实现问题，不能把这些中位差包装成统计显著性结论。

## `0.20 m -> 0.15 m` 单变量对照

| Controller | 指标 | `tol=0.20 m` | `tol=0.15 m` |
|---|---|---:|---:|
| Naive P | Success | 10/10 | 10/10 |
| Naive P | Re-departure | 10/10 | 0/10 |
| Naive P | Median Final Distance | 0.1989 m | 0.1480 m |
| Naive P | Max Post-arrival Goal Distance | 0.2028 m | 0.1698 m |
| Constrained | Success | 10/10 | 10/10 |
| Constrained | Re-departure | 10/10 | 0/10 |
| Constrained | Median Final Distance | 0.2098 m | 0.1575 m |
| Constrained | Max Post-arrival Goal Distance | 0.2112 m | 0.1717 m |

`tol=0.15 m` 后，所有 20 个 Controller-Episode 的 post-arrival 最大目标距离都留在
`0.20 m` 成功圆内，最小剩余余量约为 `0.20 - 0.1717 = 0.0283 m`。

需要注意，`Post-arrival Displacement` 从“完成 `0.20 m` 成功保持时的位置”开始计算。
现在 Controller 会继续主动向内走到约 `0.15 m` 再停，因此这项数值包含有意的向内靠近，
不应全部解释为被动停止漂移。判断本次问题是否解决，应优先看 re-departure 和
post-arrival 最大目标距离。

## 产物边界

当前 `0.15 m` 原始轨迹、进度和完整汇总：

```text
artifacts/week04/01_classical_pointgoal_benchmark/model_1500/
```

`0.20 m` 单变量对照保留在：

```text
artifacts/week04/01_classical_pointgoal_benchmark/model_1500_goal_tol020/
```

更早的停止锁存程序修复前产物保留在：

```text
artifacts/week04/01_classical_pointgoal_benchmark/model_1500_pre_stop_latch/
```

三者不合并 Episode 数；当前结论只使用 `model_1500/` 下的 `0.15 m` quick。

## 决策边界与下一步

`0.15 m` 候选已经通过静态校验、smoke 和 quick safety Gate，并解决了 quick 中的
re-departure。两个 Controller 参数与协议状态现已冻结为 `frozen`。下一步由用户启动
200 共享场景 × 2 Controller 的正式评测；当前尚未运行 `--full`。

## 复现命令

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_benchmark.py \
  --validate-only

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_benchmark.py \
  --smoke

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_benchmark.py \
  --quick
```

`--full` 的协议阶段门现已开放；正式运行仍由用户按冻结协议启动。

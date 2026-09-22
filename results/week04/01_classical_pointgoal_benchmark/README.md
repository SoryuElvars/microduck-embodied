# Week 4 Classical PointGoal Benchmark v1：正式评测记录

## 当前结论

Classical PointGoal Benchmark v1 的代码、冻结场景清单、双 Controller 运行链路、
断点续跑和 post-arrival 指标已经建立。按照单变量原则，只把两个 Controller 共享的
`goal_tolerance` 从 `0.20 m` 改为 `0.15 m`；成功半径仍为 `0.20 m`，模型、场景、seeds、
增益、速度上限、timeout 和 Gate 均未改变。

2026-09-21 完成同一组 10 个随机场景 × 2 Controller quick screen，两个 Controller
均通过 safety Gate。参数冻结后，2026-09-22 完成 200 个共享随机场景 × 2 Controller，
共 400 个正式 Controller-Episode。

正式结果中，Naive P 为 198/200 成功、2 timeout；Constrained 为 200/200 成功。两者
均为 0 fall、0 invalid state、0 post-arrival re-departure。Constrained 同时降低了路径长度、
提高了 Path Efficiency，并大幅减少 `vx/wz` 饱和和到达后的净位移，因此选为第四周胜出
Classical Controller。Naive P 的优势是多数共同成功场景完成更快，且最终停得约 1 cm
更靠近目标；这些 trade-off 在下文保留，不把 200/200 简化成所有指标都更优。

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
| Full run signature | `fa5ea0fe5726b47447262b29d7ed614d4f927a7674050f67303494d806e4a9d2` |
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

### Formal / full

Formal 使用冻结清单的全部 200 个 EpisodeSpec。每个 Episode 都固定 Start Position、
Start Yaw、Goal Position、reset seed、20 秒 timeout、`0.20 m` 成功半径、`0.5 s`
连续保持和 `2.0 s` post-arrival 观察；两个 Controller 使用同一 Episode 顺序。

| 指标 | Naive P | Constrained |
|---|---:|---:|
| 共享场景数 | 200 | 200 |
| Success | 198/200（99.0%） | 200/200（100%） |
| Success Rate Wilson 95% CI | 96.43%–99.73% | 98.12%–100% |
| Fall / Timeout / Invalid | 0 / 2 / 0 | 0 / 0 / 0 |
| Median Final Distance | 0.1488 m | 0.1588 m |
| Median absolute Final Heading Error（诊断） | 0.0148 rad | 0.0138 rad |
| Median Path Length to Arrival | 2.0143 m | 1.9192 m |
| Median Path Efficiency | 0.7280 | 0.7476 |
| Median Completion Time（成功场景） | 12.90 s | 13.29 s |
| Mean `vx` Saturation Fraction | 0.7547 | 0.0005 |
| Mean `wz` Saturation Fraction | 0.1859 | 0.1577 |
| Median Post-arrival Displacement | 0.0207 m | 0.0114 m |
| Median Post-arrival Max Displacement | 0.0257 m | 0.0276 m |
| Post-arrival Re-departure | 0/198 | 0/200 |
| Median Post-arrival Max Goal Distance | 0.1675 m | 0.1675 m |
| Maximum Post-arrival Goal Distance | 0.1714 m | 0.1717 m |

在 200 个配对场景中，Success 胜负为 Constrained 2、Naive P 0、平局 198。因为只有两组
不一致结果，精确 McNemar 双侧 `p=0.5`；因此不能只凭 99% 与 100% 声称总体成功率已有
统计显著差异。这里的胜出判断来自成功、安全、路径、饱和和停止稳定性的联合工程证据。

对两者都成功的 198 个场景，`Constrained - Naive P` 的配对统计为：

| 指标 | 配对中位差 | Constrained 更优 / Naive 更优 | 解释 |
|---|---:|---:|---|
| Final Distance（全部 200） | +0.0100 m | 2 / 198 | Naive 停得更靠近目标 |
| Path Length | -0.0358 m | 114 / 84 | Constrained 路径总体更短 |
| Path Efficiency | +0.0127 | 114 / 84 | Constrained 效率总体更高 |
| Completion Time | +0.33 s | 59 / 139 | Naive 在多数共同成功场景更快 |
| `vx` Saturation Fraction（全部 200） | -0.7602 | 200 / 0 | Constrained 大幅减少速度饱和 |
| `wz` Saturation Fraction（全部 200） | -0.0189 | 167 / 8，另 25 平局 | Constrained 多数场景更少饱和 |
| Post-arrival Displacement | -0.0093 m | 198 / 0 | Constrained 每个共同成功场景净位移都更小 |
| Post-arrival Max Displacement | +0.0018 m | 57 / 141 | 瞬时最大位移反而略大 |

Naive P 的两个 timeout 是 `w04e072 / seed 40072` 与 `w04e084 / seed 40084`。二者初始
目标相对方向分别为 `179.90°` 和 `173.68°`，都属于后方目标；最终距离分别为
`0.1703 m` 和 `0.1925 m`，已经进入 `0.20 m` 成功圈，但没有在 20 秒截止前完成连续
`0.5 s` 保持。按初始方向分桶，Naive P 在前方 44/44、侧方 110/110、后方 44/46；
Constrained 为前方 44/44、侧方 110/110、后方 46/46。

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

三者不合并 Episode 数；正式结论只使用
`model_1500/full/summary/pointgoal_full_model_1500.json`。可提交 Git 的紧凑正式统计位于
`summaries/pointgoal_full_model_1500_processed.json`，并记录原始 summary SHA-256、
协议/模型/run signature、方向分桶、分位数和逐指标配对胜负。

## 决策边界与下一步

冻结的 400 个正式 Controller-Episode 已运行完成。Constrained 在全部 200 个场景成功，
无 fall、timeout、invalid state 或 re-departure，并在路径效率、速度饱和和到达后净位移上
形成一致优势，因此确定为第四周 Classical winner。下一步只对该 Controller 运行计划内的
Low Friction、40 ms Delay、80% Motor 轻量 OOD；不得把 Naive P 的 formal、quick、smoke
或第三周结果混入其 OOD 成功率。

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

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_benchmark.py \
  --full

cd ~/projects/microduck-embodied
python3 evaluation/analyze_pointgoal_benchmark.py
```

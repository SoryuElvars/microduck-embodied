# MicroDuck 400-Episode 随机速度基准

日期：2026-09-18

## 协议

- 模型：`model_1500.onnx`
- Episode：400
- Reset seeds：42–441
- Command seed：20260911
- 指令范围：`vx [-0.4, 0.4]`、`vy [-0.3, 0.3]`、`wz [-1.0, 1.0]`
- 显式评测配额：25% 静止、15% 原地转向、60% 全范围均匀采样
- 该配额来自最终训练配置中的 standing/turn buckets，但为便于分桶统计而设为互斥；不是对训练采样器执行顺序的逐项复刻
- 每个 Episode：1 秒零指令预热 + 10 秒测试，官方 reset，CPU MuJoCo + BAM M6
- 部署形态 ONNX 评测不加载 Reward Manager，因此 `Episode Return` 不可用

## 与固定八指令矩阵的关系

此前的固定矩阵包含 8 条轴向隔离指令 × 20 seeds，共 160 个 Episode；它适合
诊断前进、后退、横移和转向的单项缺陷。本次 400 个 Episode 覆盖同时出现的
vx/vy/wz 组合以及显式 standing/turn buckets。两套协议互补，结果分别报告，
不合并为一个成功率，也不把重复子集再次计数。

## 与 PointGoal 接口的边界

该随机协议覆盖正负 `vx`、非零 `vy` 和全范围 `wz`，检验的是完整三轴速度命令
覆盖。当前第一阶段 PointGoal Navigator 固定输出 `[vx, 0, wz]`，因此三轴联合
Success 不是 PointGoal 到达率，也不能覆盖或推翻独立的闭环 PointGoal 结果。
其中 `vx/wz` 响应、零命令停止行为和 Fall Rate 与当前任务直接相关；非零 `vy`
结果记录为更广 locomotion 能力边界。

## 数据完整性

- 汇总 Episode：400
- 原始 CSV：400
- CSV 总行数：220000
- 每个 CSV 字段数：95
- 缺失文件：0
- 行数不一致：0
- 非有限数值：0

## 项目定义的 Nominal Success

这不是上游官方指标。本项目把“未跌倒，且每个轴的 RMSE 不超过该轴命令幅值
的 30%”定义为一次 nominal success。为避免零命令被数值噪声误判，vx/vy/wz
分别使用 0.03 m/s、0.03 m/s、0.10 rad/s 的最小容差。这个混合门槛也能避免
把低速指令完全不响应错误地计为成功。

- 运动指令 Nominal Success Rate：**0.0%**（排除静止 Episode）
- 全部 Episode Nominal Success Rate：0.0%（包含 100 个静止 Episode）
- Fall Rate：**0.0%**
- vx / vy / wz 单轴通过率：48.0% / 0.0% / 4.2%

## Episode 均值响应诊断

下面使用相同容差检查“Episode 实际均值与命令的偏差”，用于区分长期响应偏置与
Episode 内步态振荡。它是看到结果后补充的诊断，不是预先冻结的 Nominal Success，
不能取代上面的逐步速度 RMSE Gate。`vx/wz` 列只对应当前 PointGoal Navigator 的
`[vx, wz]` 接口边界，也不是新的通过门槛。

| 分组 | Episodes | 三轴 Episode-mean pass | `vx/wz` Episode-mean pass |
|---|---:|---:|---:|
| general | 240 | 1.2% | 16.7% |
| standing | 100 | 100.0% | 100.0% |
| turn_in_place | 60 | 100.0% | 100.0% |
| motion_only | 300 | 21.0% | 33.3% |
| overall | 400 | 40.8% | 50.0% |

三轴 Episode-mean 单轴通过率为：
`vx 50.5% / vy 47.2% / wz 89.2%`。

## 分桶结果

| Bucket | Episodes | Success Rate | Fall Rate | Mean RMSE vx | Mean RMSE vy | Mean RMSE wz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| general | 240 | 0.0% | 0.0% | 0.102 | 0.170 | 0.437 |
| standing | 100 | 0.0% | 0.0% | 0.015 | 0.066 | 0.225 |
| turn_in_place | 60 | 0.0% | 0.0% | 0.018 | 0.106 | 0.262 |

## 指令响应

| 轴 | 响应斜率 | 相关系数 | Episode 均值绝对误差 |
| --- | ---: | ---: | ---: |
| vx | 0.521 | 0.998 | 0.062 m/s |
| vy | 0.170 | 0.967 | 0.071 m/s |
| wz | 1.040 | 0.980 | 0.078 rad/s |

原地转向的分方向结果：

- 正向：命令均值 +0.681，实际均值 +0.742 rad/s，成功率 0.0%
- 负向：命令均值 -0.677，实际均值 -0.740 rad/s，成功率 0.0%

100 个静止 Episode 与 300 个运动指令必须分开解释；运动指令
成功率为 0.0%，不能用包含静止样本的
0.0% 代替控制器的运动跟踪结论。

## 代表性运动样本

- 最佳合格运动样本：无。
- 典型失败：Episode 10，seed 52，命令 `(-0.158, +0.140, +0.095)`，RMSE `(0.067, 0.160, 0.230)`。
- 最严重失败：Episode 165，seed 207，命令 `(+0.373, -0.137, -0.194)`，RMSE `(0.171, 0.153, 0.820)`。

## 结论

固定模型的运动指令均未通过严格的逐轴 RMSE 门槛。即使保持零跌倒，也不能据此冻结为具有完整随机速度跟踪能力的控制器。

当前第一阶段 PointGoal 接口继续固定为 `[vx,0,wz]`，非零 `vy` 作为范围外能力
单独记录。该边界允许候选进入 PointGoal OOD 快筛，但停止漂移、`vx` 欠跟踪和
yaw 振荡必须作为风险指标持续报告；不能把任务范围收窄写成完整 locomotion 通过。

源汇总：`artifacts/week03/06_yaw_only_tracking_candidate/random_velocity_400/seed42_model_1500/summary/random_velocity_400_model_1500.json`

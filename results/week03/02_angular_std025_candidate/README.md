# Angular tracking std 单变量候选 A

日期：2026-09-14

## 当前状态

训练前配置、测试、CUDA 冒烟、25-iteration 预运行、4096-env 正式训练和
2000-iteration Checkpoint Gate 均已完成。本实验保留 `model_5999` 作为失败
Control，候选 A 从头训练，只收紧 `track_angular_velocity.std`。候选 A 的
2000-iteration Gate 未通过；由于该点刚进入最终 curriculum 且训练指标仍在
改善，下一步保持所有参数不变，诊断续训至 3000 iterations。本候选尚未通过，
也不会直接续训到 4000/6000。

## 基线溯源

| 项目 | 值 |
| --- | --- |
| 原始训练 commit | `29e887ecfbf5d37144759e5a9f8a176dfb83d547` |
| 原始训练工作区 | clean，保存的 Git diff 为空 |
| Control checkpoint | `model_5999.pt` |
| Control seed / envs | `42 / 4096` |
| Control angular std | `sqrt(0.5)` |
| 候选开发基线 | 本地 `develop@53b8971` |
| 候选分支 | `codex/angular-std-025` |

`29e887e..53b8971` 只增加视觉和公寓场景文件，未修改 velocity 训练
配置。本次不基于已包含 symmetry 续训代码的分支，也不引入远端
`develop` 后续的机器人常量变更。

## 唯一训练变量

```text
Control:      track_angular_velocity.std = sqrt(0.5)  ~= 0.7071
Candidate A:  track_angular_velocity.std = sqrt(0.25) = 0.5000
```

保持不变：Reward weight、command sampling、Domain Randomization、PPO 参数、
training seed、环境数量和 Observation/Action 合约。不启用 mirror loss。

## 候选 Task 与默认门禁

```text
Task:             Mjlab-Velocity-Flat-Angular-Std025-MicroDuck
experiment_name:  velocity_angular_std025
run_name:         angular-std025-seed42-from-scratch
seed:             42
resume:           false
save_interval:    250
max_iterations:   2000
```

默认在 2000 iterations 停止，不会自动继续到 4000/6000。

## 训练顺序

### 1. CUDA 冒烟测试

```bash
cd ~/projects/microduck_rl

uv run train Mjlab-Velocity-Flat-Angular-Std025-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.run-name angular-std025-seed42-smoke-64x5
```

### 2. 25-iteration 预运行

```bash
cd ~/projects/microduck_rl

uv run train Mjlab-Velocity-Flat-Angular-Std025-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 25 \
  --agent.run-name angular-std025-seed42-prerun-64x25
```

### 3. 正式训练到 2000-iteration Gate

```bash
cd ~/projects/microduck_rl

uv run train Mjlab-Velocity-Flat-Angular-Std025-MicroDuck \
  --env.scene.num-envs 4096
```

训练前两步只用于拒绝配置错误、NaN、明显跌倒和 Reward 异常，不用
64-env 小批结果判定最终策略质量。

## Checkpoint Gate

在 `500 / 1000 / 1500 / 2000` checkpoints 分别导出 ONNX，并用
[`angular_std025_gate_5.json`](../../../evaluation/command_sets/angular_std025_gate_5.json)
运行三指令 × 5 seeds 配对快筛：

```text
vx=+0.3
wz=+0.5
wz=-0.5
```

同时检查速度 RMSE、直行净偏航、正负转向方向、Fall Rate、NaN 和
Action 平滑性。只有 2000-iteration Gate 通过后才续训到 4000/6000。
本次 2000 Gate 未通过后的 3000-iteration 延长仅用于判断最终 curriculum 下
是否仍在收敛，不视为通过长期训练门禁。

## 验证记录

- [x] 候选 Task 已注册。
- [x] 候选与基线环境配置只有 angular std 不同。
- [x] Runner 从头训练、seed 42、symmetry 关闭。
- [x] `64 envs x 5 iterations` CUDA 冒烟测试。
- [x] `64 envs x 25 iterations` 预运行。
- [x] `4096 envs x 2000 iterations` 正式训练。
- [x] 导出并校验 `500 / 1000 / 1500 / 1999` 四个 ONNX。
- [x] 每个 checkpoint 完成三指令 x 5 seeds 配对快筛。

### 2026-09-14 冒烟与预运行结果

| 项目 | `64 x 5` 冒烟 | `64 x 25` 预运行 |
| --- | ---: | ---: |
| 最终 iteration | 4 | 24 |
| 最终 checkpoint | `model_4.pt` | `model_24.pt` |
| 中位 iteration 耗时 | - | `0.935 s` |
| 中位吞吐 | - | `1643 steps/s` |
| 非有限 TensorBoard 标量 | 0 | 0 |
| `nan_state` 最大值 | 0 | 0 |

两个 Run 的保存配置均为 `64 envs`、seed 42、`std=0.5`、
`resume=false`。预运行早期仍频繁跌倒且 Episode 较短，符合从随机策略开始的
25-iteration 现象；该结果只通过配置可运行与 NaN 拒绝门禁，不作为策略
质量结论。

### 2026-09-14 正式训练与 Checkpoint Gate

正式训练使用 `4096 envs`、seed 42 和 `std=0.5`。训练在约 750 iterations
临时停止，随后从 `model_750.pt` 恢复至最终 `model_1999.pt`；checkpoint 中的
`common_step_counter` 连续增长，curriculum 没有因恢复训练而重置。由于迭代
从 0 编号，计划中的 2000-iteration checkpoint 对应 `model_1999.pt`。

四个 PT checkpoint 均通过官方 `scripts/export.py` 导出，ONNX checker、有限
输出和 `Observation 61 -> Action 14` 合约均通过。每个 checkpoint 独立运行
3 条指令 x 5 seeds，共 60 Episode；以下均为 10 秒测试段的五 Episode 均值：

| Checkpoint | 前进 `vx` / RMSE | 前进 `wz` / 净偏航 | `wz=+0.5` 实际值 / RMSE | `wz=-0.5` 实际值 / RMSE | Fall Rate | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `model_500` | `+0.163 / 0.139` | `+0.073 / +41.1 deg` | `+0.067 / 0.558` | `-0.042 / 0.612` | `0%` | 未通过 |
| `model_1000` | `+0.208 / 0.101` | `-0.142 / -80.7 deg` | `+0.267 / 0.351` | `-0.195 / 0.395` | `0%` | 未通过 |
| `model_1500` | `+0.155 / 0.149` | `-0.394 / -228.0 deg` | `+0.320 / 0.375` | `-0.395 / 0.303` | `0%` | 未通过 |
| `model_1999` | `+0.208 / 0.101` | `-0.235 / -134.7 deg` | `+0.005 / 0.496` | `-0.003 / 0.497` | `0%` | 未通过 |

完整的跨 checkpoint 聚合指标、配对 seeds、Policy SHA256、非有限数检查和
Action 变化统计见
[`checkpoint_gate_500_1999.json`](summaries/checkpoint_gate_500_1999.json)。

![500 至 1999 checkpoint 三指令 Gate 对比](figures/checkpoint_gate_500_1999.png)

最终 checkpoint 的三指令响应如下。图中的目标转向为 `±0.5 rad/s`，实际
响应接近零；前进指令下则存在明显负向 yaw。

![model_1999 三指令响应](figures/model_1999_command_response.png)

所有原始 CSV 数值均为有限值。测试段相邻 Action 的 L2 变化均值从
`0.478`（500）下降到 `0.141`（1999），但动作更平滑没有转化为合格的命令
跟踪。

`model_1000` 和 `model_1500` 已能按正负命令给出正确转向方向，但直行偏航
持续恶化；`model_1999` 的原地正负转向又退化到接近零响应，同时直行仍有
显著负向偏航。该结果与训练过程中偏低的 `track_angular_velocity` 一致，表明
它不是仅由训练指标尺度造成的观感问题，而是闭环策略质量缺陷。

结论：候选 A 在 2000 iterations 时不满足正式续训门槛。考虑到
`standing_envs` 和 `head_pose_range` 的最终 curriculum 阶段在 2000 附近才
生效，且 W&B 训练指标仍未进入平台，下一步保持 std、seed 和其余配置不变，
诊断续训至 3000，并在约 2500 和 3000 重复相同 Gate。该延长不代表当前
checkpoint 已通过；如果闭环转向仍接近零或直行偏航没有改善，则停止候选。

## 图表复现

跨 checkpoint 汇总和主图由
[`plot_checkpoint_gate.py`](../../../evaluation/plot_checkpoint_gate.py) 从被 Git
忽略的 suite JSON 与原始 CSV 生成；最终 checkpoint 图复用
[`plot_command_response.py`](../../../evaluation/plot_command_response.py)。依赖
官方环境执行：

```bash
cd ~/projects/microduck_rl

uv run python ~/projects/microduck-embodied/evaluation/plot_checkpoint_gate.py \
  --suite ~/projects/microduck-embodied/artifacts/week03/02_angular_std025_candidate/model_500/summary/angular_std025_gate_5_model_500.json \
  --suite ~/projects/microduck-embodied/artifacts/week03/02_angular_std025_candidate/model_1000/summary/angular_std025_gate_5_model_1000.json \
  --suite ~/projects/microduck-embodied/artifacts/week03/02_angular_std025_candidate/model_1500/summary/angular_std025_gate_5_model_1500.json \
  --suite ~/projects/microduck-embodied/artifacts/week03/02_angular_std025_candidate/model_1999/summary/angular_std025_gate_5_model_1999.json \
  --summary-output ~/projects/microduck-embodied/results/week03/02_angular_std025_candidate/summaries/checkpoint_gate_500_1999.json \
  --figure-output ~/projects/microduck-embodied/results/week03/02_angular_std025_candidate/figures/checkpoint_gate_500_1999.png

uv run python ~/projects/microduck-embodied/evaluation/plot_command_response.py \
  --suite ~/projects/microduck-embodied/artifacts/week03/02_angular_std025_candidate/model_1999/summary/angular_std025_gate_5_model_1999.json \
  --output ~/projects/microduck-embodied/results/week03/02_angular_std025_candidate/figures/model_1999_command_response.png
```

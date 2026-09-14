# Angular tracking std 单变量候选 A

日期：2026-09-14

## 当前状态

训练前配置、测试、CUDA 冒烟和 25-iteration 预运行已完成；
4096-env 正式训练尚未启动。本实验保留
`model_5999` 作为失败 Control，候选 A 从头训练，只收紧
`track_angular_velocity.std`。

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

## 验证记录

- [x] 候选 Task 已注册。
- [x] 候选与基线环境配置只有 angular std 不同。
- [x] Runner 从头训练、seed 42、symmetry 关闭。
- [x] `64 envs x 5 iterations` CUDA 冒烟测试。
- [x] `64 envs x 25 iterations` 预运行。
- [ ] `4096 envs x 2000 iterations` 正式训练。

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

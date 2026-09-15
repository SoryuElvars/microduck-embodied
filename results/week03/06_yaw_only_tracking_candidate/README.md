# Candidate B：yaw-only angular tracking 训练前准备

日期：2026-09-15

## 决策依据

`model_2500` 的 25-Episode Classical PointGoal pilot 为 20/25 到达、0% 跌倒，
但左侧目标 0/5、右侧目标 5/5。随后 20-Episode 低速 yaw 归因显示：

- `vx=0.05, wz=+0.50` 的实际 `wz` 约为 `+0.011`，响应率约 2.3%；
- `vx=0.05, wz=-0.50` 的实际 `wz` 约为 `-0.361`，响应率约 72.2%；
- 提高到 `vx=0.10` 后正 yaw 响应有所恢复，但仍明显弱于负 yaw。

因此下一轮训练只验证一个假设：原 angular tracking 奖励把 yaw error 与机身
roll/pitch angular velocity 合并，会使行走中的横向角速度干扰期望的 yaw 跟踪；
将正奖励改为只度量 commanded yaw rate 与 body-frame yaw rate 的误差，是否能够
减轻低速正 yaw 启动死区和左右不对称。

## 唯一行为修改

Control（Candidate A）的 angular tracking error：

```text
e = (wz_cmd - wz_actual)^2 + wx_actual^2 + wy_actual^2
r = exp(-e / std^2)
```

Candidate B：

```text
e = (wz_cmd - wz_actual)^2
r = exp(-e / std^2)
```

保持不变：

- `std = sqrt(0.25) = 0.5`
- angular tracking reward weight
- command sampling 与 curriculum
- PPO 参数、seed 42、4096 envs 正式训练规模
- 从头训练，不从 Candidate A checkpoint 续训
- 不加入 mirror loss 或 controller 调参

roll/pitch angular velocity 仍由独立的 `body_ang_vel` penalty 约束。

## 官方仓库实现

- 仓库：`/home/elvars/projects/microduck_rl`
- 分支：`codex/yaw-only-angular-tracking`
- Task：`Mjlab-Velocity-Flat-Yaw-Only-Tracking-MicroDuck`
- Experiment：`velocity_yaw_only_tracking`
- Run：`yaw-only-tracking-seed42-from-scratch`
- 计划 checkpoint：`500 / 1000 / 1500 / 1999`，必要时再按证据续训

实现新增 yaw-only reward callable，并基于 Candidate A 复制环境和 runner 配置；
仅替换 `track_angular_velocity.func`。独立 Task/Experiment/Run 名称用于避免与旧模型
日志混淆，不属于训练行为变化。

## 验证结果

### 单元与配置差异测试

```text
uv run --with pytest pytest -q \
  tests/test_velocity_angular_tracking_candidate.py \
  tests/test_velocity_yaw_only_tracking_candidate.py
```

结果：`5 passed`。

测试覆盖：

- 完美 yaw 跟踪时，即使 `wx/wy` 非零，yaw-only reward 仍为 1；
- yaw error 为 0.5、std 为 0.5 时，reward 为 `exp(-1)`；
- Candidate A 与 B 的环境配置只在 angular reward callable 上不同；
- Candidate B 继承 seed、PPO、save interval、max iterations 和禁用 symmetry 等设置。

### CUDA 冒烟与预运行

| 门禁 | 结果 | 本地日志 | W&B run |
|---|---|---|---|
| 64 envs × 5 iterations | 通过 | `logs/rsl_rl/velocity_yaw_only_tracking/2026-09-15_22-42-01_yaw-only-tracking-seed42-from-scratch` | `sn2sj9xq` |
| 64 envs × 25 iterations | 通过 | `logs/rsl_rl/velocity_yaw_only_tracking/2026-09-15_22-43-31_yaw-only-tracking-seed42-from-scratch` | `7wuldkwe` |

两次均使用 WSL2 CUDA、NVIDIA RTX 4060 Laptop GPU、BAM M6，actor observation
为 61 维、action 为 14 维。25-iteration 末轮：

- `nan_state = 0`
- Mean reward `0.55`
- Mean episode length `37.28`
- Mean action std `0.97`
- Mean value loss `0.0328`
- Mean surrogate loss `-0.0385`

`model_4.pt`、`model_24.pt` 和对应 ONNX 均已落盘。短预运行从随机初始化开始，
仍频繁跌倒是预期现象；这里的通过仅表示配置、CUDA、Reward 数值、训练与导出链路
没有明显异常，不表示策略已学会稳定行走。

## 当前结论与下一步

Candidate B 已通过正式训练前的工程门禁，但尚无策略质量证据，也尚未开始
4096-env 长训。下一步应先提交官方仓库这组独立改动，再由用户启动从头训练。
到达 checkpoint 后必须沿用 PointGoal 核心指令和配对 seeds 做快筛；只有 yaw
对称性、直行偏航和速度跟踪出现目标改善且稳定性不退化，才复测同一个
25-Episode PointGoal pilot。

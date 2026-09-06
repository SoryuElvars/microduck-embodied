# MicroDuck Embodied

基于 MicroDuck 的具身算法学习与项目实现，围绕腿式机器人强化学习、鲁棒运动控制、目标点导航和分层策略展开。

当前目标是在六周内完成一个可复现、可量化、可用于实习投递的项目版本：

```text
Goal
  ↓
Navigation
  ↓
Velocity Command
  ↓
Robust RL Locomotion
  ↓
MicroDuck
```

## 项目目标

- 跑通官方 MicroDuck Locomotion 的训练、回放、ONNX 导出与推理链路。
- 建立可重复运行的 Locomotion Benchmark。
- 评测摩擦、控制延迟、电机强度和回程间隙等 Reality Gap 因素。
- 实现传统 Go-to-Goal Controller。
- 实现冻结底层步态策略的 RL Navigator。
- 在常规及 OOD 环境中比较 Classical 与 RL Navigation。
- 使用定量指标、失败案例、图表和演示视频呈现实验结果。

## 三阶段主线

### M1：Robust Locomotion

复现并理解官方 PPO Locomotion，建立基线评测，并开展 Reality Gap 敏感性与鲁棒性实验。

### M2：Goal-conditioned Navigation

以目标相对位姿为输入，由高层导航模块输出速度指令，底层继续使用 Locomotion Policy；分别实现传统控制器和 RL Navigator。

### M3：Task-level Skill Composition

在 M1、M2 稳定后，进一步组合 Navigate、Align、Kick、Recover 等技能，完成“将球移动到目标区域”的任务。M3 不属于当前前六周范围。

## 当前六周路线图

| 周次 | 核心任务 | 主要成果 |
|---|---|---|
| 第 1 周 | 官方链路复现 | Train → Play → ONNX → Inference |
| 第 2 周 | Locomotion Benchmark | 速度跟踪、成功率、跌倒率和典型轨迹 |
| 第 3 周 | 鲁棒性实验 | Baseline vs Robust，Reality Gap 敏感性曲线 |
| 第 4 周 | Classical PointGoal | 随机 Start/Goal 自主到达 |
| 第 5 周 | RL Navigator | Frozen Locomotion + High-level PPO |
| 第 6 周 | 统一对比与整理 | Classical vs RL、OOD、README、图表和视频 |

完整执行清单见：[MicroDuck 前六周实施计划](./MicroDuck_前六周实施计划.md)。

## 官方 Locomotion 基线

当前复现任务为 [`Mjlab-Velocity-Flat-MicroDuck`](https://github.com/pollen-robotics/microduck_rl)。该任务的目的不只是让 MicroDuck“会往前走”，而是训练一个在平地上运行的通用低层运动策略：

- 接收机器人坐标系下的线速度与角速度指令 $[v_x, v_y, \omega]$；
- 接收头部姿态指令，在行走时保持头部可控；
- 以 50 Hz 输出 14 个伺服关节的连续控制动作；
- 在质心、摩擦、电机、观测噪声和控制延迟等随机化下学习稳定步态；
- 导出包含 Observation Normalizer 的 ONNX 策略，为后续 Sim2Real 和真机部署做准备；
- 作为 PointGoal Navigation 的冻结低层控制器，由上层 Navigator 输出速度指令。

因此，官方基线训练验证的是“指令跟踪 + 稳定性 + 动作质量 + 鲁棒性”，不能只通过总 Reward 判断是否成功。

### 本次复现记录

| 项目 | 值 |
|---|---|
| 官方仓库 | `pollen-robotics/microduck_rl` |
| 官方 Commit | [`29e887ec`](https://github.com/pollen-robotics/microduck_rl/commit/29e887ecfbf5d37144759e5a9f8a176dfb83d547) |
| 任务 | `Mjlab-Velocity-Flat-MicroDuck` |
| 随机种子 | `42` |
| 并行环境数 | `4096` |
| 总训练量 | `6000 iterations` |
| 续训点 | `model_5000.pt` |
| 最终 Checkpoint | `model_5999.pt` |
| W&B Runs | `baseline-flat-4096x6000` + `baseline-flat-resume-5000` |

完整的命令、环境、产物、问题复现和第一周验收结论见：[第 1 周官方 Locomotion 链路复现记录](./results/week01_official_locomotion_chain.md)。

### W&B 必看图表

查看续训曲线时，应同时选中上述两个 Run，将 X 轴设为 `Step`，不要将它们 Group 成两次独立实验。续训 Run 在 iteration 5000 附近的 Reward 和 Episode Length 瞬时降低是 episode 统计重置造成的记录边界，不代表策略崩溃。

| 优先级 | 图表 | 观察重点 |
|---|---|---|
| P0 | `Train/mean_reward` | 总体上升并趋于稳定；不能单独作为成功证据 |
| P0 | `Train/mean_episode_length` | 行走任务中应逐渐接近 1000 steps / 20 s |
| P0 | `Metrics/twist/error_vel_xy` | 平面线速度指令跟踪误差，越低越好 |
| P0 | `Metrics/twist/error_vel_yaw` | 偏航角速度跟踪误差，越低越好 |
| P0 | `Episode_Termination/fell_over` | 因姿态倾覆而结束，趋势应下降 |
| P0 | `Episode_Termination/nan_state` | 物理状态异常，必须始终为 0 |
| P0 | `Episode_Metrics/mean_action_acc` | 动作的离散二阶变化，越低通常越平滑 |
| P1 | `Episode_Reward/track_linear_velocity` | 线速度跟踪正奖励，应上升并稳定 |
| P1 | `Episode_Reward/track_angular_velocity` | 角速度跟踪正奖励，应上升并稳定 |
| P1 | `Episode_Reward/upright` | 躯干直立奖励，过低可能表示长期倾斜 |
| P1 | `Episode_Reward/air_time` | 合理脚部腾空时间，用于促进交替迈步 |
| P1 | `Episode_Reward/head_pose_tracking` | 头部姿态指令跟踪能力 |
| P1 | `Policy/mean_std` | 连续动作探索强度，应逐渐降低但不应过早趋近 0 |
| P1 | `Episode_Reward/action_rate_l2` | 动作变化惩罚，必须不大于 0，用于抑制抖动 |
| P1 | `Episode_Reward/foot_slip` | 支撑脚滑动惩罚，应接近 0 |
| P1 | `Episode_Reward/self_collisions` | 机器人自碰撞惩罚，应接近 0 |
| P1 | `Episode_Reward/dof_pos_limits` | 关节接近极限惩罚，应接近 0 |
| P2 | `Loss/value` | Critic 回报预测误差，关注是否稳定，不要要求严格单调下降 |
| P2 | `Loss/surrogate` | PPO Actor 优化目标对应的损失，通常在 0 附近波动 |
| P2 | `Loss/entropy` | 策略分布的探索相关项，需与 `Policy/mean_std` 一起观察 |
| P2 | `Loss/learning_rate` | 自适应 PPO 的实际学习率，上下波动本身正常 |
| P2 | `Curriculum/*` | 训练难度、指令范围和惩罚权重，是解释曲线台阶变化的上下文 |
| P2 | `Perf/total_fps` | 训练吞吐率，只反映运行效率，不反映策略质量 |

解读规则：

- `Episode_Reward/<term>` 是乘过权重的贡献值；不同 Reward 项不能只按数值大小直接比较。
- 所有惩罚项必须不大于 0，否则可能存在权重符号错误。
- `Episode_Reward/body_pose_tracking` 在当前 Velocity 任务中权重为 0，图表为 0 不代表失败。
- Curriculum 切换附近允许 Reward 短暂下降，但应在后续训练中恢复。
- 总 Reward 上升但速度跟踪项没有改善，可能是策略只学会了站立或利用其他奖励。

### 6000 Iterations 后的验证门禁

W&B 只能证明训练过程基本健康，不能单独证明步态合格。最终策略必须依次通过以下四层验证。

### 本次验证结论（2026-09-06）

本次运行已经完成训练、Checkpoint、ONNX 导出和 CPU MuJoCo 推理，工程链路判定为通过；但 `model_5999.pt` 未通过纯前进质量检查：

| 固定命令 | 观测结果 |
|---|---|
| `vx=+0.30 m/s, vy=0, wz=0` | 第 2 秒实际 `yaw_rate=+0.250 rad/s` |
| 同一命令持续 10 秒 | 累计航向约 `+166.5°`，形成稳定单侧弧线 |
| 实际前进速度 | 约 `0.18 m/s`，低于 `0.30 m/s` 命令 |

训练曲线中的 `Metrics/twist/error_vel_yaw` 从约 2000 iterations 起进入平台期，直到 6000 iterations 仍维持在约 `1.06～1.15`。该 W&B 指标是按 episode 累计、归一化后的趋势指标，不能直接视为瞬时 `rad/s`。

显式导出的 ONNX 与自动导出的 ONNX 已在 100 组随机观测上完成对比，最大绝对输出差为 `0.0`。因此当前证据指向训练策略本身的方向偏置，而不是 ONNX 损坏或两条导出链路不一致。

本模型保留为失败 Baseline，不作为后续导航系统中冻结的合格底层策略。

#### 1. Checkpoint 回放

在官方仓库中执行：

```bash
cd ~/projects/microduck_rl

RUN_DIR=logs/rsl_rl/velocity/2026-09-05_22-08-33_baseline-flat-resume-5000

uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "$RUN_DIR/model_5999.pt" \
  --viewer native
```

若 WSLg 的原生窗口无法打开，将 `--viewer native` 改为 `--viewer viser`。回放时至少观察：是否长时间直立、是否真正交替迈步、是否频繁拖脚或打滑、转向时是否失稳。

#### 2. 显式导出 ONNX

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "$RUN_DIR/model_5999.pt" \
  --onnx-file "$RUN_DIR/microduck_velocity_flat_5999.onnx"
```

只使用官方导出入口，确保 Observation Normalizer 被写入 ONNX。

#### 3. CPU MuJoCo 部署形态推理

```bash
uv run scripts/infer_policy.py \
  --walking "$RUN_DIR/microduck_velocity_flat_5999.onnx" \
  --new-cmd-obs
```

推理窗口中使用终端键盘分别验证：

- `Space`：零速度指令下站立；
- `↑` / `↓`：前进与后退；
- `←` / `→`：左右横移；
- `A` / `E`：左右转向；
- `H`：切换头部指令模式，检查行走中的头部跟踪；
- `P`：施加随机推力，只记录策略的抗扰与失败表现，不要将跌倒后自主站起作为 Velocity 任务的要求。

ONNX 策略应与 Checkpoint 回放呈现一致的主要行为，且终端打印的 achieved/cmd 速度应能随指令方向正确变化。

#### 4. 固定 Benchmark

完成视觉验收后，进入第 2 周，在固定配置和随机种子下运行 200～500 个 Episode，至少产出：

- $RMSE_{v_x}$、$RMSE_{v_y}$ 和 $RMSE_{\omega}$；
- Fall Rate、Nominal Success Rate 和 Episode Return；
- Target Velocity vs Actual Velocity 曲线；
- 典型成功、失败和转向失稳案例；
- Checkpoint、官方 Commit、评测种子和配置的完整记录。

正式 Benchmark 完成前，对当前策略的结论应表述为“官方 PPO 工程链路已完成，但当前策略未通过直线行走质量验收”，不宣称其已经具备合格的速度跟踪或可量化鲁棒性。

## 计划中的目录结构

```text
microduck-embodied/
├── configs/       # 训练与实验配置
├── evaluation/    # Locomotion、鲁棒性和导航评测
├── navigation/    # Classical Controller 与 RL Navigator
├── robot/         # RobotState、Backend 等抽象接口
├── scripts/       # 训练、评测和导出入口
├── results/       # 汇总数据、图表和实验说明
└── tests/         # 接口与核心逻辑测试
```

目录将在项目推进过程中按实际需求逐步建立，避免提前堆积空目录。

## 评测指标

Locomotion：

- Velocity Tracking RMSE
- Command Tracking Error
- Episode Return
- Fall Rate
- Success Rate

Navigation：

- Success Rate
- Final Position Error
- Path Length
- Completion Time
- Fall Rate

## 官方工作与个人贡献

| 官方项目提供 | 本项目重点实现 |
|---|---|
| MicroDuck 模型 | 自动 Benchmark 框架 |
| Locomotion 任务 | Reality Gap 与鲁棒性评测 |
| PPO Locomotion 基线 | Robot / Backend 抽象接口 |
| 训练及推理运行时 | Classical PointGoal Navigator |
|  | RL High-level Navigator |
|  | Classical vs RL 统一对比 |

本项目不会将官方已有的机器人模型、Locomotion 任务或 PPO 基线声明为个人原创成果。后续将在确定所使用的官方仓库与版本后，补充来源链接和 Commit 信息。

## 当前状态

第 1 周工程门禁已完成：WSL2 + CUDA 环境、`Mjlab-Velocity-Flat-MicroDuck` 4096 并行环境训练、Checkpoint、ONNX Export 和 CPU MuJoCo Inference 均已跑通，最终 Checkpoint 为 `model_5999.pt`。

当前进入第 2 周 Locomotion Benchmark。由于该策略在零转向纯前进命令下出现稳定偏航，它只作为失败 Baseline 保留；下一步先建立固定、可重复的评测，再开展第二轮训练配置实验。

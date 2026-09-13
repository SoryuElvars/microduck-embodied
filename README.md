# MicroDuck Embodied

基于 MicroDuck 的具身算法学习与项目实现，围绕腿式机器人强化学习、鲁棒运动控制、
目标点导航和分层策略展开。

项目主线：

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

以目标相对位姿为输入，由高层导航模块输出速度指令，底层继续使用 Locomotion Policy；
分别实现传统控制器和 RL Navigator。

### M3：Task-level Skill Composition

在 M1、M2 稳定后，进一步组合 Navigate、Align、Kick、Recover 等技能。M3 不属于当前
前六周范围。

## 六周路线图

| 周次 | 核心任务 | 主要成果 |
|---|---|---|
| 第 1 周 | 官方链路复现 | Train → Play → ONNX → Inference |
| 第 2 周 | Locomotion Benchmark | 速度跟踪、成功率、跌倒率和典型轨迹 |
| 第 3 周 | 鲁棒性实验 | Baseline vs Robust，Reality Gap 敏感性曲线 |
| 第 4 周 | Classical PointGoal | 随机 Start/Goal 自主到达 |
| 第 5 周 | RL Navigator | Frozen Locomotion + High-level PPO |
| 第 6 周 | 统一对比与整理 | Classical vs RL、OOD、README、图表和视频 |

完整执行清单见：[MicroDuck 前六周实施计划](./MicroDuck_前六周实施计划.md)。

## 官方 Locomotion 任务

底层运动策略基于官方
[`Mjlab-Velocity-Flat-MicroDuck`](https://github.com/pollen-robotics/microduck_rl)，其通用功能是：

- 接收机器人坐标系下的线速度与角速度指令 $[v_x, v_y, \omega]$；
- 接收头部和躯干姿态指令；
- 以 50 Hz 输出 14 个主动伺服关节的连续控制动作；
- 通过随机化与 BAM 执行器模型学习稳定、可部署的步态；
- 导出包含 Observation Normalizer 的 ONNX Actor，作为上层导航系统的底层控制器。

官方任务提供机器人、环境、PPO 训练和推理运行时。本项目不复制这些实现，而是在独立
仓库中建立评测、鲁棒性实验和导航系统。

## 仓库边界与目录

```text
/home/elvars/projects/
├── microduck_rl/          # 官方训练、环境、模型与推理
└── microduck-embodied/    # 个人评测、结果、导航与展示
    ├── evaluation/       # 评测、分析和绘图脚本
    ├── results/          # 可提交的汇总、图表和报告
    ├── artifacts/        # 本地原始数据与中间产物
    └── tests/            # 评测器与分析逻辑测试
```

## 评测原则

- 训练完成、总 Reward 高或不跌倒，都不能单独证明策略合格。
- 同时报告 Velocity Tracking RMSE、Command Tracking Error、Fall Rate 和 Nominal Success Rate。
- ONNX 部署评测默认不含训练 Reward；Episode Return 由官方 PT checkpoint + Reward Manager 评测。
- 固定轴向指令、随机组合指令和官方环境 Return 是独立协议，不合并 Episode 数或成功率。
- 所有评测记录 checkpoint、上游 commit、配置、随机种子、Episode 数和时长。

## 官方工作与个人贡献

| 官方项目提供 | 本项目重点实现 |
|---|---|
| MicroDuck 模型与 Locomotion 任务 | 自动 Benchmark 框架 |
| PPO Locomotion 基线 | Reality Gap 与鲁棒性评测 |
| 训练、导出与推理运行时 | Robot / Backend 抽象接口 |
|  | Classical PointGoal Navigator |
|  | RL High-level Navigator |
|  | Classical vs RL 统一对比 |

本项目不将官方机器人模型、Locomotion 任务或 PPO 基线声明为个人原创成果。

## 阶段记录

根 README 只保留长期通用信息。具体复现环境、run、checkpoint、实验数值和验收结论按阶段维护：

- [第 1 周：官方 Locomotion 链路复现记录](./results/week01_official_locomotion_chain.md)
- [第 2 周：Locomotion 自动评测与训练诊断](./results/week02/)

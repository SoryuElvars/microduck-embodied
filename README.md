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

项目处于初始化阶段，当前主要依据六周计划推进 M1 与 M2。实验结果、架构图和演示材料将在完成相应里程碑后持续补充。


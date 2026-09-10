# MicroDuck 具身算法项目前六周实施计划

> 目标：在 6 周内完成一个可写入简历、可用于投递具身算法/机器人强化学习实习的项目版本。
>
> 项目主线：**官方 Locomotion 基线 → 鲁棒性评测 → PointGoal Navigation → Classical vs RL 对比**。

## 1. 六周结束时的目标成果

六周后，项目至少应包含：

- 跑通的官方 `microduck_rl` 训练、回放、ONNX 导出与推理链路；
- 一套可重复运行的 Locomotion 自动评测框架；
- 一组面向 Reality Gap 的敏感性与鲁棒性实验；
- 一个基于传统控制器的 PointGoal Navigation 基线；
- 一个冻结底层步态策略、只训练高层导航器的分层 RL 系统；
- Classical 与 RL Navigation 在常规环境和 OOD 环境中的定量对比；
- 完整 README、系统架构图、定量图表和 30～60 秒演示视频；
- 可以直接用于简历描述和面试讲解的实验结论。

最终能力链：

```text
MuJoCo
  + PPO
  + Legged Locomotion
  + Domain Randomization / Reality Gap Evaluation
  + Goal-conditioned Navigation
  + Hierarchical RL Architecture
```

## 2. 总体进度表

| 周次 | 核心任务 | 必须完成的结果 | 对简历的价值 |
|---|---|---|---|
| 第 1 周 | 跑通官方 `microduck_rl` 全链路 | Train → Play → ONNX Export → Inference | 证明掌握官方 RL 技术栈 |
| 第 2 周 | 理解并评测 Locomotion | Observation/Action/Reward 梳理 + Benchmark | 从“运行 Demo”升级为“分析策略” |
| 第 3 周 | Reality Gap / 鲁棒性实验 | Friction、Delay、Motor Strength、Backlash | 形成 Sim2Real-ready 实验能力 |
| 第 4 周 | PointGoal + 传统控制 Baseline | 随机 Start/Goal 条件下自主到达 | 完成第一个自己的核心模块 |
| 第 5 周 | RL Navigator | PPO：Goal → 速度指令 | 形成分层 RL 系统 |
| 第 6 周 | 对比实验与项目整理 | Classical vs RL + OOD + README + 视频 | 达到可正式投递的项目版本 |

里程碑：

- **第 3 周结束：** M1 已形成可展示结果，开始整理简历第一版。
- **第 4 周结束：** M2 Classical 版本跑通，开始投递第一批实习。
- **第 5 周结束：** RL Navigator 有稳定结果，更新简历。
- **第 6 周结束：** 完成正式展示版，扩大投递范围。

---

## 3. 第 1 周：跑通官方完整链路

### 本周目标

不修改算法，先完整跑通官方流程：

```text
microduck_rl
    ↓
训练 PPO
    ↓
保存 Checkpoint
    ↓
Play
    ↓
导出 ONNX
    ↓
infer_policy
```

### 任务清单

- [x] 完成项目环境与依赖安装。
- [x] 跑通官方 `Mjlab-Velocity-Flat-MicroDuck` 任务（4096 environments，训练至 iteration 5999）。
- [x] 成功保存并加载 Checkpoint（最终为 `model_5999.pt`）。
- [x] 在 MuJoCo 中播放训练后的行走策略，并完成纯前进人工检查。
- [x] 将策略导出为 ONNX 模型。
- [x] 使用导出的 ONNX 模型完成推理。
- [ ] 若时间充足，再运行 `Mjlab-VelStand-Flat-MicroDuck`。
- [x] 建立自己的项目仓库和结果目录，不把所有实验直接堆在官方仓库中。

### 必须弄清楚的问题

- [x] Observation 包含哪些信息？（当前 ONNX 输入为 61 维观测。）
- [x] Policy 输出什么？（14 个关节的连续动作。）
- [x] Action 如何转换为舵机或关节命令？（默认姿态与动作缩放形成位置目标，再由 BAM M6 执行器模型产生控制。）
- [x] Reward 如何引导机器人学会行走？（已结合 W&B 分项 Reward 与速度误差进行检查。）
- [x] Policy 的运行频率是多少？（50 Hz，MuJoCo timestep 0.005 s、decimation 4。）

### 建议的项目目录

```text
microduck_embodied/
├── configs/
├── evaluation/
├── navigation/
├── robot/
├── scripts/
└── results/
```

### 本周交付物

- [x] Training Curve（W&B：主训练与 5000 次后的续训 Run）
- [ ] Walking Video
- [x] Checkpoint（`model_5999.pt`，仅保留在本地官方仓库日志中）
- [x] ONNX Model（`microduck_velocity_flat_5999.onnx`，仅保留在本地）
- [x] 一份官方链路运行记录，包括命令、配置、结果路径和遇到的问题：[`results/week01_official_locomotion_chain.md`](./results/week01_official_locomotion_chain.md)

### 验收标准

能够独立完成：

```text
训练 → 加载 Checkpoint → 仿真行走 → 导出 ONNX → ONNX 推理
```

> 阶段门禁：如果第 1 周结束时仍卡在环境安装或官方流程，不进入 M2，先把完整链路补齐。

### 第 1 周复盘（2026-09-06）

- **工程链路：已完成。** 环境、训练、Checkpoint、ONNX 导出和 CPU MuJoCo 推理均已跑通。
- **策略质量：未通过直线行走检查。** 在 `vx=+0.30 m/s, vy=0, wz=0` 下，第 2 秒实际偏航角速度约 `+0.250 rad/s`，10 秒累计航向约 `+166.5°`。
- **训练趋势：已进入失败平台。** `Metrics/twist/error_vel_yaw` 从约 2000 iterations 起长期维持在约 `1.06～1.15`，追加训练到 6000 没有解决问题。
- **导出链路：排除为主要原因。** 自动导出与显式导出的 ONNX 在 100 组随机输入下最大输出差为 `0.0`。
- **阶段判断：允许进入第 2 周。** 当前模型作为失败 Baseline 保留；第 2 周先把直行、转向和跌倒检查固化为可重复 Benchmark，再进行第二轮训练配置实验。
- **未完成的非阻塞交付物：** Walking Video；待获得具有代表性的可展示策略后补充。

完整记录见：[`results/week01_official_locomotion_chain.md`](./results/week01_official_locomotion_chain.md)。

---

## 4. 第 2 周：建立官方 Locomotion Benchmark

### 本周目标

把官方 Locomotion 从“可运行 Demo”变成一套可重复、可量化的基线。

### 任务清单

- [x] 编写自动评测脚本：[`evaluation/locomotion_benchmark.py`](./evaluation/locomotion_benchmark.py)。
- [ ] 自动随机生成速度指令：
  - $v_x$
  - $v_y$
  - $\omega$
- [ ] 连续运行 200～500 个 Episode。
- [x] 固定评测配置和随机种子，确保实验可以复现。
- [x] 自动保存逐 Episode 原始数据和汇总结果。

### 必须统计的指标

- [x] $RMSE_{v_x}$
- [x] $RMSE_{v_y}$
- [x] $RMSE_{\omega}$
- [x] Fall Rate
- [ ] Episode Return
- [x] Command Tracking Error（当前以三轴 RMSE 和平均实际速度表示）
- [ ] Nominal Success Rate
- [ ] 若 Torque 数据容易获取，再增加 Energy Proxy；否则暂缓。

### 必须生成的图表

- [x] Target Velocity vs Actual Velocity
- [ ] 典型成功轨迹
- [ ] 典型失败轨迹或失稳案例

当前 8 项固定指令、每项 20 Episode 的响应矩阵见：
[`results/week02/command_response_20.md`](./results/week02/command_response_20.md)。

### 系统理解与说明材料

整理清楚以下内容，并形成文档或示意图：

- [ ] Observation
- [ ] Action
- [ ] Reward
- [ ] Control Frequency
- [ ] Policy Architecture

建议系统图：

```text
Velocity Command
       ↓
Observation
       ↓
PPO Policy
       ↓
Joint Action
       ↓
MicroDuck
       ↓
State Feedback
```

### 本周交付物

- [ ] `Official Locomotion Baseline Report`
- [ ] 自动评测脚本
- [ ] 原始评测数据
- [ ] 汇总指标表
- [ ] Target vs Actual 速度跟踪曲线
- [ ] Locomotion 系统架构图

### 验收标准

报告至少包含：

- Nominal Success Rate
- Velocity Tracking Error
- Fall Rate
- 若干典型轨迹
- 对 Observation、Action、Reward 和控制频率的清晰解释

---

## 5. 第 3 周：Reality Gap 与鲁棒性实验

### 本周目标

完成真正的敏感性测试，并比较 Baseline Policy 与 Robust Policy 在相同 OOD 条件下的表现。

### 必做实验参数

| 不确定因素 | 推荐测试值 |
|---|---|
| Ground Friction | 0.3 / 0.5 / 0.7 / 0.9 / 1.1 |
| Control Delay | 0 / 20 / 40 / 60 / 80 ms |
| Motor Strength | 80% / 90% / 100% / 110% |
| Backlash | Normal / Backlash Model |

有余力时再增加：

- [ ] Mass ±10%
- [ ] IMU Noise
- [ ] Joint Encoder Noise

### 任务清单

- [ ] 为四类必做不确定因素建立统一的配置入口。
- [ ] 使用固定测试集分别评测每一个参数档位。
- [ ] 记录每次实验的配置、随机种子、Checkpoint 和结果路径。
- [ ] 生成敏感性曲线，例如 Delay vs Fall Rate、Friction vs Success Rate。
- [ ] 训练或使用一个带 Domain Randomization 的 Robust Policy。
- [ ] 在完全相同的 OOD 测试条件下比较 Baseline Policy 与 Robust Policy。
- [ ] 分析 MicroDuck 对哪类 Reality Gap 最敏感。

### 对比表模板

| Test Condition | Baseline SR | Robust SR | 主要现象 |
|---|---:|---:|---|
| Nominal |  |  |  |
| Low Friction |  |  |  |
| 40 ms Delay |  |  |  |
| 80% Motor Strength |  |  |  |
| Backlash |  |  |  |

### 本周交付物

- [ ] 四类不确定因素的完整实验数据
- [ ] Baseline vs Robust 汇总表
- [ ] 2～4 张敏感性或对比曲线
- [ ] Reality Gap 影响分析
- [ ] 典型 Failure Cases

### 验收标准

至少完成：

```text
4 类 Uncertainty
    +
Baseline vs Robust
    +
2～4 张曲线
    +
结果分析
```

> 完成本周后结束 M1，不继续无限调步态，立即转入 M2。

---

## 6. 第 4 周：PointGoal Navigation 与传统控制基线

### 本周目标

建立第一个真正属于自己的算法模块：随机 Start/Goal 条件下的自主目标点导航。

### 任务定义

高层接收：

$$
[\Delta x,\ \Delta y,\ \Delta yaw]
$$

高层输出：

$$
[v_x,\ v_y,\ \omega]
$$

底层继续调用官方 Locomotion Policy：

```text
Goal
 ↓
Go-to-Goal Controller
 ↓
vx, vy, ω
 ↓
Official PPO Locomotion
 ↓
MicroDuck
```

第一版建议只使用：

```text
vx + yaw rate
```

暂时令 $v_y=0$，降低双足机器人控制和调试难度。

### 软件接口边界

建立清晰的数据和模块接口：

- [ ] `RobotState`
- [ ] `GoalState`
- [ ] `VelocityCommand`
- [ ] `Navigator`
- [ ] `MujocoBackend`

禁止让 Navigator 直接读取 `mj_data.qpos`。应采用：

```text
MujocoBackend
      ↓
RobotState
      ↓
Navigator
```

这样未来迁移真机时，只需替换状态来源和底层执行后端。

### 传统控制器

距离与朝向误差：

$$
d=\sqrt{\Delta x^2+\Delta y^2}
$$

$$
\alpha=\operatorname{atan2}(\Delta y,\Delta x)
$$

基本控制律：

$$
v=k_d d
$$

$$
\omega=k_\alpha \alpha
$$

必须增加：

- [ ] Velocity Limit
- [ ] Acceleration Limit
- [ ] Goal Tolerance
- [ ] 到达目标后的减速和停止逻辑

### 评测任务

- [ ] 随机 Start Position
- [ ] 随机 Start Yaw
- [ ] 随机 Goal Position
- [ ] 运行至少 200 个 Episode

统计：

- [ ] Success Rate
- [ ] Final Position Error
- [ ] Path Length
- [ ] Completion Time
- [ ] Fall Rate

### 本周交付物

- [ ] Robot / Goal / Command 抽象接口
- [ ] Go-to-Goal Controller
- [ ] PointGoal 自动评测脚本
- [ ] 200 个 Episode 的评测结果
- [ ] 随机 Start/Goal 自主导航视频

### 验收标准

点击运行后不再需要人工摇杆输入，机器人可以：

```text
随机出生
   ↓
收到 Goal
   ↓
自主转向
   ↓
自主行走
   ↓
接近目标时减速
   ↓
停在目标容差范围内
```

> 本周结束时形成“构建 MicroDuck 分层目标导航框架”的第一版可展示成果，并开始正式投递第一批实习。

---

## 7. 第 5 周：训练 RL Navigator

### 本周目标

保持底层 Locomotion Policy 冻结，只训练高层 Navigation Policy。

系统结构：

```text
High-level PPO
      ↓
Velocity Command
      ↓
Frozen Low-level PPO
      ↓
MicroDuck
```

### 状态与动作设计

高层策略：

$$
\pi_{nav}: o_t \rightarrow (v_x,v_y,\omega)
$$

建议 Observation：

$$
o_t=[\Delta x,\Delta y,\Delta\theta,v_x,v_y,\omega]
$$

建议 Action：

$$
a_t=[v_x^*,v_y^*,\omega^*]
$$

第一版推荐简化为：

$$
a_t=[v_x^*,\omega^*],\qquad v_y=0
$$

### Reward 设计

保持简单，只使用四类主要奖励：

$$
r=r_{progress}+r_{goal}+r_{smooth}+r_{fall}
$$

进度奖励：

$$
r_{progress}=d_{t-1}-d_t
$$

到达奖励：

$$
r_{goal}=+R
$$

动作平滑惩罚：

$$
r_{smooth}=-\lambda\lVert a_t-a_{t-1}\rVert^2
$$

摔倒惩罚：

$$
r_{fall}=-R_f
$$

### 任务清单

- [ ] 冻结 M1 Low-level Locomotion Policy。
- [ ] 定义高层 Observation 和 Action Space。
- [ ] 完成 PointGoal Navigation 训练环境。
- [ ] 实现四项简洁 Reward。
- [ ] 随机化 Start 和 Goal。
- [ ] 记录训练曲线、成功率和失败原因。
- [ ] 使用与传统控制器一致的评测接口。
- [ ] 不在本周扩展成包含大量奖励项的复杂 Reward。

### 本周交付物

- [ ] RL Navigator 训练代码与配置
- [ ] 可加载的 Navigation Policy Checkpoint
- [ ] 训练曲线
- [ ] 随机 Start/Goal 的独立评测结果
- [ ] 典型成功与失败视频

### 验收标准

RL Navigator 能在随机 Start/Goal 环境中稳定收敛，并能通过统一 Benchmark 重复评测。

> 不提前写死必须达到的成功率数值；先确保训练稳定、评测一致、结果可复现，再根据 Classical Baseline 确定合理目标。

---

## 8. 第 6 周：完成可投递版本

### 本周目标

不继续堆新功能，集中完成统一对比、结果分析、项目说明和展示材料。

### 8.1 Classical vs RL 统一对比

在同一个固定测试集上分别运行：

- Go-to-Goal Controller
- RL Navigator

测试规模：

```text
500 Random Episodes
```

常规环境对比表：

| Metric | Classical | RL |
|---|---:|---:|
| Success Rate |  |  |
| Position Error |  |  |
| Path Length |  |  |
| Completion Time |  |  |
| Fall Rate |  |  |

OOD 环境至少包括：

- [ ] Low Friction
- [ ] 40 ms Control Delay
- [ ] Motor Weakness

实验要求：

- [ ] 两种方法使用完全相同的测试 Episode、随机种子和终止条件。
- [ ] 同时展示总体结果与典型 Failure Case。
- [ ] 回答“为什么导航需要 RL”以及“传统方法在什么条件下更合适”。

### 8.2 整理 README

README 建议结构：

```text
1. Project Overview
2. System Architecture
3. Official MicroDuck Baseline
4. Robust Locomotion Evaluation
5. Goal-conditioned Navigation
6. Classical vs RL
7. Sim2Real-ready Design
8. Demo
9. Limitations
10. Future Work
```

README 中必须清楚区分：

| Official Work | My Contribution |
|---|---|
| MicroDuck Model | Benchmark Framework |
| Locomotion Task | Reality-gap Evaluation |
| PPO Baseline | Robot Abstraction |
| Runtime | Goal Navigator |
|  | RL High-level Policy |
|  | Classical vs RL Comparison |

### 8.3 准备简历与展示素材

- [ ] 一张完整系统架构图：

```text
Goal
 ↓
Navigation
 ↓
Velocity Command
 ↓
RL Locomotion
 ↓
MicroDuck
```

- [ ] 一段 30～60 秒 Demo 视频，至少展示随机 Start、随机 Goal 和自主到达。
- [ ] 2～4 张定量图表，例如：
  - Latency vs Fall Rate
  - Friction vs Success Rate
  - Baseline vs Robust
  - Classical vs RL
- [ ] 一个结构清楚、README 完整的 GitHub 仓库。
- [ ] 一版简历项目描述。
- [ ] 一份面试讲解提纲，覆盖问题定义、方法、指标、结果、失败案例和个人贡献。

### 本周验收标准

- [ ] Classical 与 RL 已在同一测试集上完成 500 个随机 Episode 对比。
- [ ] 常规与 OOD 结果均有定量表格和结论。
- [ ] README 能让陌生人理解项目目标、架构、结果和个人贡献。
- [ ] Demo、图表和简历项目描述齐全。
- [ ] 项目达到可以公开展示和正式投递的状态。

---

## 9. 六周内的范围边界

前六周只完成 M1 + M2，暂不进入 M3。以下内容全部推迟到第 7 周以后：

- VLA / VLM
- Camera Perception
- SLAM
- M3 Skill Library
- Kick Task
- LLM
- 自行设计机器人本体
- 复杂 Terrain Planning

六周内只守住这一条主线：

```text
Goal
  ↓
Navigation
  ↓
Velocity Command
  ↓
Robust RL Locomotion
```

原则：

- M1 的目标是理解、复现和评测官方能力，不长期沉迷于调整步态 Reward。
- M2 是第一个主要个人贡献，应优先保证架构清楚、指标可信和结果可复现。
- 每新增一个功能前，先确认它是否直接服务于六周目标和实习投递。
- 失败案例和限制必须保留，不只展示剪辑后的成功 Demo。
- 不把官方已有内容写成个人原创贡献。

---

## 10. 实验记录规范

每次正式实验至少记录：

```text
Experiment ID
Date
Git Commit
Task / Environment
Config
Random Seed
Policy Checkpoint
Number of Episodes
Metrics
Result Directory
Notes / Failure Cases
```

建议结果目录：

```text
results/
├── week01_official_pipeline/
├── week02_locomotion_baseline/
├── week03_robustness/
├── week04_classical_navigation/
├── week05_rl_navigation/
└── week06_comparison_and_demo/
```

每周结束时完成一次复盘：

- [ ] 本周计划完成了什么？
- [ ] 哪些指标已经有可靠结果？
- [ ] 最主要的失败原因是什么？
- [ ] 是否产生了可放入 README 或简历的材料？
- [ ] 下周是否满足阶段门禁？
- [ ] 是否出现范围膨胀，需要删减任务？

---

## 11. 总验收清单

### M1：Locomotion 与鲁棒性

- [x] 官方训练、Play、ONNX 导出和推理全链路跑通（策略质量缺陷另行记录）
- [ ] Locomotion Benchmark 可自动运行
- [ ] Nominal 指标与速度跟踪曲线齐全
- [ ] 四类 Reality Gap 实验完成
- [ ] Baseline vs Robust 对比完成
- [ ] 有定量曲线、失败案例和结果分析

### M2：Goal-conditioned Navigation

- [ ] RobotState / GoalState / VelocityCommand / Navigator 接口清楚
- [ ] Classical Go-to-Goal Baseline 完成
- [ ] 随机 Start/Goal 的 200 Episode 评测完成
- [ ] Frozen Locomotion + RL Navigator 完成
- [ ] Classical vs RL 的 500 Episode 统一对比完成
- [ ] OOD 对比完成

### 投递材料

- [ ] README
- [ ] 系统架构图
- [ ] 2～4 张定量图表
- [ ] 30～60 秒 Demo
- [ ] 简历项目描述
- [ ] 面试讲解提纲
- [ ] GitHub 仓库结构与实验结果整理完毕

## 12. 完成定义

本阶段不是以“仿真中偶尔成功一次”为完成，而是以以下条件为准：

1. 方法可以通过脚本重复运行；
2. 结果来自固定 Benchmark 和足够数量的 Episode；
3. 有 Baseline、有对比、有失败案例；
4. 能明确解释官方工作与个人贡献；
5. README、图表、视频和简历材料共同形成完整证据链。

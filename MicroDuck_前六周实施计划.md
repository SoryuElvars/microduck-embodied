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
| 第 3 周 | 底层模型选择、训练与鲁棒性实验 | 官方模型与自训练候选完成 Nominal/OOD 对比并冻结导航底层 | 形成 Sim2Real-ready 实验能力 |
| 第 4 周 | PointGoal + 传统控制对比 | Naive P 与受限 Go-to-Goal 在随机 Start/Goal 下统一评测 | 完成第一个自己的核心模块 |
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
- [x] 自动随机生成速度指令：
  - $v_x$
  - $v_y$
  - $\omega$
- [x] 连续运行 200～500 个 Episode（已完成 400 个随机指令 Episode）。
- [x] 固定评测配置和随机种子，确保实验可以复现。
- [x] 自动保存逐 Episode 原始数据和汇总结果。

### 必须统计的指标

- [x] $RMSE_{v_x}$
- [x] $RMSE_{v_y}$
- [x] $RMSE_{\omega}$
- [x] Fall Rate
- [x] Episode Return（使用官方 PT checkpoint + Reward Manager，完成 8 指令 × 5 个成对初始状态）
- [x] Command Tracking Error（当前以三轴 RMSE 和平均实际速度表示）
- [x] Nominal Success Rate（项目定义门槛；与上游官方指标区分）
- [ ] 若 Torque 数据容易获取，再增加 Energy Proxy；否则暂缓。

### 必须生成的图表

- [x] Target Velocity vs Actual Velocity
- [x] 典型成功轨迹（当前仅站立静止成功；不宣称存在运动成功样本）
- [x] 典型失败轨迹或失稳案例（随机基准 Episode 322 和 362）

当前 8 项固定指令、每项 20 Episode 的响应矩阵见：
[`results/week02/01_baseline_5999/README.md`](./results/week02/01_baseline_5999/README.md)。

### 系统理解与说明材料

整理清楚以下内容，并形成文档或示意图：

- [x] Observation
- [x] Action
- [x] Reward
- [x] Control Frequency
- [x] Policy Architecture

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

- [x] [`Official Locomotion Baseline Report`](./results/week02/01_baseline_5999/README.md)
- [x] 自动评测脚本
- [x] 原始评测数据（本地 `artifacts/week02/`）
- [x] 汇总指标表
- [x] Target vs Actual 速度跟踪曲线
- [x] Locomotion 系统架构图

### 验收标准

报告至少包含：

- Nominal Success Rate
- Velocity Tracking Error
- Fall Rate
- 若干典型轨迹
- 对 Observation、Action、Reward 和控制频率的清晰解释

---

## 5. 第 3 周：底层模型选择、训练与鲁棒性实验

### 本周目标

在进入导航开发前，评测官方推理模型、训练自己的单变量候选模型，并在统一的
Nominal/OOD 协议下选择一个可冻结的底层 Locomotion Policy。第三周完成的是
控制器级 Reality Gap 敏感性实验；没有真机数据时，结论应标记为
`Sim2Real proxy`，不得写成已经完成真实 Reality Gap 验证。

候选模型分为：

- 已有失败基线：自训练 `model_5999`；
- 官方参考模型：固定 Hugging Face revision 和 SHA-256 的
  `alpha_walking.onnx`；
- 自训练候选 A：只将 angular-velocity tracking std 收紧到
  `sqrt(0.25)` 的新模型；
- 自训练候选 B：仅当候选 A 保持稳定但仍压不住偏航时，再将
  std 进一步收紧到 `sqrt(0.1)`。

`model_5999` 用于保留失败基线，不再为它投入与优胜候选相同规模的完整 OOD
预算。官方模型是否用于第四周，必须由本项目的统一评测决定，不能因“官方”身份
免除验收。

### 5.1 官方模型接入与 Nominal 评测

- [x] 固定官方模型 revision、SHA-256、下载来源和本地结果路径。
- [x] 验证 ONNX 的 `Observation 61 → Action 14` 合约。
- [x] 核对 50 Hz 控制频率、Observation 排列和 command encoding。
- [x] 运行固定八指令 × 5 seeds 快筛，并完成 MJCF、Normalizer、执行器和运行时兼容性排查。
- [ ] 快筛通过后，运行固定八指令 × 20 seeds 正式对比。
- [ ] 运行独立的 400-Episode 随机速度指令协议。
- [ ] 运行 Action 镜像诊断，记录正负转向和横移的不对称性。

固定八指令、随机速度指令和官方 PT Reward Manager Return 继续作为不同协议
分别报告，不合并 Episode 数或成功率。

### 5.2 自训练候选模型

第一轮只验证一个假设：当前转向不对称和直行偏航是否主要来自角速度跟踪奖励
过宽。`track_angular_velocity` 对机身三轴角速度误差使用 Gaussian
奖励；直接从 `sqrt(0.5)` 降到 `sqrt(0.1)` 会将指数惩罚放大 5 倍，
训练早期存在大误差样本奖励接近零的风险。因此先使用中间值：

```text
Control:      track_angular_velocity.std = sqrt(0.5)
Candidate A:  track_angular_velocity.std = sqrt(0.25)
Candidate B:  track_angular_velocity.std = sqrt(0.1)  # A 稳定但偏航仍不合格时再训练

保持不变：reward weight、command sampling、PPO 参数、seed、环境数量
暂不叠加：mirror loss 或其他新参数
```

训练顺序：

- [x] 为候选 A 建立独立 Task/Run 名称，保留原始 baseline。
- [x] 用测试锁定新旧环境配置只有 angular tracking std 不同。
- [x] 运行 `64 envs × 5 iterations` CUDA 冒烟测试。
- [x] 运行 `64 envs × 25 iterations` 预运行，检查 NaN、跌倒和 Reward 异常。
- [ ] 使用 4096 envs 从头训练，按固定间隔保存 checkpoint。
- [ ] 在 `500 / 1000 / 1500 / 2000` iterations 进行三指令 × 5 seeds 快筛。
- [ ] 只有 2000-iteration Gate 通过后，才继续到 `4000 / 6000`。
- [ ] 若候选 A 稳定但偏航仍未达标，再建立独立候选 B；不在同一
  Run 中修改 std。
- [ ] 候选有效后，再增加 1～2 个独立 training seeds 判断训练随机性；不得用
  evaluation seeds 代替 training seeds。

训练中使用的三条核心诊断指令：

```text
vx=+0.3
wz=+0.5
wz=-0.5
```

### 5.3 必做鲁棒性参数

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

### 5.4 控制器级鲁棒性评测

- [ ] 为四类必做不确定因素建立统一配置入口。
- [ ] 所有候选先用静止、前进、正负转向和前进加转向做 OOD 快筛。
- [ ] 只对 Nominal 表现最好的两个模型运行完整四因素矩阵。
- [ ] 所有模型共享指令、seeds、终止条件和参数档位。
- [ ] 记录配置、模型 revision/checkpoint、种子和结果路径。
- [ ] 统计 Tracking RMSE、方向成功率、直行偏航、Fall Rate、恢复时间和停止漂移。
- [ ] 生成 Delay/Friction/Motor Strength/Backlash 敏感性曲线。
- [ ] 分析 MicroDuck 对哪类 Reality Gap 因素最敏感。

400-Episode 随机速度指令只代表 Nominal 指令覆盖；若物理参数没有改变，不能将
它单独称为鲁棒性测试。

### 对比表模板

| Test Condition | `model_5999` | Official | Self-trained Candidate | 主要现象 |
|---|---:|---:|---:|---|
| Nominal |  |  |  |  |
| Low Friction |  |  |  |  |
| 40 ms Delay |  |  |  |  |
| 80% Motor Strength |  |  |  |  |
| Backlash |  |  |  |  |

### 本周交付物

- [ ] 官方 `alpha_walking.onnx` 独立评测报告
- [ ] angular tracking 单变量训练报告
- [ ] `model_5999` / 官方模型 / 自训练候选的 Nominal 对比表
- [ ] 四类不确定因素的统一配置入口和完整实验数据
- [ ] 最优两个模型的鲁棒性汇总表
- [ ] 2～4 张敏感性或对比曲线
- [ ] Reality Gap / Sim2Real proxy 影响分析
- [ ] 典型 Failure Cases
- [ ] 一个冻结供第四周使用的 Locomotion Policy

### 验收标准

导航候选至少满足：

- 正负转向方向均正确，且不是单个 checkpoint 的偶然行为；
- 直行偏航显著低于 `model_5999`；
- 低速命令存在可预测响应，静止保持稳定；
- Nominal 条件无明显跌倒；
- OOD 退化能够量化，且不是轻微扰动下立即全面失效。

至少完成：

```text
官方模型评测
    +
自训练单变量候选
    +
4 类 Uncertainty
    +
最优两个底层模型统一对比
    +
冻结第四周底层策略
```

若官方模型通过门禁，可先冻结官方模型进入第四周；自训练模型仍作为研究和替换
候选。若所有模型均未通过，第四周可先实现接口与控制器单元测试，但不得把使用
不合格底层得到的 PointGoal 结果写成最终导航结论。

> 完成本周后结束 M1 的主要实验，不继续无限调步态，立即转入 M2。

---

## 6. 第 4 周：PointGoal Navigation 与传统控制对比

### 本周目标

使用第三周冻结的底层策略，建立随机 Start/Goal 条件下的自主目标点导航，并形成
一个可由第五周 RL Navigator 直接复用的 Classical Navigation Benchmark。

这里的 Go-to-Goal Controller 本身就是传统控制基线。第四周先比较简单比例控制
与加入工程约束的实用控制器；真正的 Classical Navigator vs RL Navigator 统一
对比仍放在第五周训练完成后的第六周。

### 6.1 任务定义

高层接收：

$$
[\Delta x,\ \Delta y,\ \Delta yaw]
$$

高层输出：

$$
[v_x,\ v_y,\ \omega]
$$

底层调用第三周冻结的 Locomotion Policy：

```text
Goal
 ↓
Classical Navigator
 ↓
vx, vy, ω
 ↓
Frozen Locomotion Policy
 ↓
MicroDuck
```

第一版只使用：

```text
vx + yaw rate
vy = 0
```

避免在第一版同时引入横移跟踪缺陷，降低双足机器人控制和调试难度。

### 6.2 软件接口边界

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

Navigator 与具体 ONNX 解耦。未来迁移真机或替换底层策略时，只替换状态来源和
执行后端，不重写导航算法。

### 6.3 传统控制器 A：Naive P

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

Naive P 只增加必要的速度限幅，作为最简单、可解释的传统基线。

### 6.4 传统控制器 B：Constrained Go-to-Goal

在相同误差定义上增加：

- [ ] Velocity Limit
- [ ] Acceleration Limit
- [ ] Goal Tolerance
- [ ] 大航向误差时优先转向并抑制前进速度
- [ ] 接近目标时连续减速
- [ ] 到达目标后的停止保持
- [ ] 指令低通或变化率限制
- [ ] 超时、跌倒和失败判定

第四周首先回答：加入这些工程约束是否相对 Naive P 提高成功率、减少过冲并改善
停止稳定性。

### 6.5 底层策略替换对比

如果第三周同时得到合格的官方模型和自训练模型，则保持 Constrained
Go-to-Goal 的所有参数不变，只替换底层 ONNX：

```text
Constrained Go-to-Goal + Official Locomotion
                       vs
Constrained Go-to-Goal + Self-trained Locomotion
```

该实验用于隔离底层控制质量对导航的影响，报告名称应为
`Official vs Self-trained Low-level Policy under the same Classical Navigator`。
不得将它称为 Classical vs RL Navigator，因为两组使用的高层仍是传统控制器。

### 6.6 评测任务

开发阶段先使用固定 Goal：

- [ ] 正前方
- [ ] 左前方
- [ ] 右前方
- [ ] 后方
- [ ] 不同初始 yaw

逻辑通过后冻结 controller 参数，再运行：

- [ ] 随机 Start Position
- [ ] 随机 Start Yaw
- [ ] 随机 Goal Position
- [ ] 至少 200 个共享 Episode

统计：

- [ ] Success Rate
- [ ] Final Position Error
- [ ] Final Yaw Error
- [ ] Path Length / Path Efficiency
- [ ] Completion Time
- [ ] Fall Rate
- [ ] Timeout Rate
- [ ] 到达后的停止漂移

两种传统控制器及不同底层模型组合必须使用完全相同的 Start/Goal、seeds、终止
条件和成功门槛。

### 6.7 导航系统轻量 OOD 验证

第四周不重复第三周的完整四因素敏感性矩阵，只对最终 Classical Controller 选择
三个代表条件运行小规模闭环测试：

- [ ] Low Friction
- [ ] 40 ms Control Delay
- [ ] 80% Motor Strength

这一步验证底层鲁棒性是否能够转化为 Goal 到达能力。完整的 Classical vs RL
常规/OOD 统一对比仍保留到第六周。

### 本周交付物

- [ ] Robot / Goal / Command / Navigator 抽象接口
- [ ] Naive P Controller
- [ ] Constrained Go-to-Goal Controller
- [ ] PointGoal 自动评测脚本
- [ ] 两种传统控制器的统一对比表
- [ ] 必要时完成官方 vs 自训练底层模型替换对比
- [ ] 200 个随机 Start/Goal 的冻结参数评测结果
- [ ] 轨迹、位置误差和完成时间图表
- [ ] 典型成功与失败案例
- [ ] 随机 Start/Goal 自主导航视频

### 验收标准

点击运行后不再需要人工摇杆输入，机器人可以：

```text
随机出生
   ↓
收到 Goal
   ↓
自主调整航向
   ↓
自主行走
   ↓
接近目标时减速
   ↓
停在目标容差范围内
```

同时要求：

- Constrained Go-to-Goal 相对 Naive P 有明确改善，或提供无改善的定量解释；
- 结果来自冻结参数后的统一 200-Episode 测试；
- 传统控制器与底层 ONNX 解耦；
- 第五周可直接复用相同接口和 Benchmark 训练 RL Navigator。

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

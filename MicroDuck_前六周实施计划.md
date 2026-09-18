# MicroDuck 具身算法项目前六周实施计划

> 目标：在 6 周内完成一个可写入简历、可用于投递具身算法/机器人强化学习实习的项目版本。
>
> 项目主线：**官方 Locomotion 基线 → 轻量 PointGoal 闭环诊断 → 证据驱动的底层训练与鲁棒性评测 → PointGoal Navigation → Classical vs RL 对比**。

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
| 第 3 周 | 底层筛选、轻量 PointGoal pilot、单变量再训练与鲁棒性实验 | 用闭环失败证据确定训练因素，完成新旧候选复测并冻结导航底层 | 形成从任务诊断到策略改进的实验闭环 |
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

## 5. 第 3 周：底层筛选、轻量 PointGoal pilot 与证据驱动训练

### 本周目标

在进入第四周正式导航开发前，先对已有 Locomotion 候选做指令级快筛，再用
一个 `20～30 Episode` 的 Classical PointGoal pilot 检查它在闭环导航中的真实失败
模式。根据 pilot 证据选择一个训练因素，从头训练新候选；新旧模型通过
同一 Locomotion 协议和同一 PointGoal pilot 复测后，才对优胜者运行完整
Nominal/OOD 对比并冻结第四周底层策略。

第三周的 PointGoal pilot 是选择底层模型的轻量诊断，不是第四周的
Classical Navigator 正式 Benchmark，不与后者合并 Episode 数或成功率。
第三周完成的鲁棒性结论仍是控制器级 `Sim2Real proxy`；没有真机数据时，
不得写成已经完成真实 Reality Gap 验证。

候选模型分为：

- 已有失败基线：自训练 `model_5999`；
- 官方参考模型：固定 Hugging Face revision 和 SHA-256 的
  `alpha_walking.onnx`；
- 已完成的自训练候选 A：只将 angular-velocity tracking std 收紧到
  `sqrt(0.25)`，当前以 `model_2500` 作为 pilot 临时候选；
- 已完成 2000-iteration 正式训练的自训练候选 B：根据 PointGoal pilot 和低速
  yaw 归因，只把 angular tracking 改为 yaw-axis error；已完成快筛与
  同协议 PointGoal pilot，但尚未满足冻结条件。

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

### 5.2 已完成的 angular std 单变量候选

第一轮只验证一个假设：当前转向不对称和直行偏航是否主要来自角速度跟踪奖励
过宽。`track_angular_velocity` 对机身三轴角速度误差使用 Gaussian
奖励；直接从 `sqrt(0.5)` 降到 `sqrt(0.1)` 会将指数惩罚放大 5 倍，
训练早期存在大误差样本奖励接近零的风险。因此先使用中间值：

```text
Control:      track_angular_velocity.std = sqrt(0.5)
Candidate A:  track_angular_velocity.std = sqrt(0.25)
保持不变：reward weight、command sampling、PPO 参数、seed、环境数量
暂不叠加：mirror loss 或其他新参数
```

训练顺序：

- [x] 为候选 A 建立独立 Task/Run 名称，保留原始 baseline。
- [x] 用测试锁定新旧环境配置只有 angular tracking std 不同。
- [x] 运行 `64 envs × 5 iterations` CUDA 冒烟测试。
- [x] 运行 `64 envs × 25 iterations` 预运行，检查 NaN、跌倒和 Reward 异常。
- [x] 使用 4096 envs、seed 42 从头训练至 `model_1999.pt`，保持配置不变续训至
  `model_2999.pt`。
- [x] 导出 `500 / 1000 / 1500 / 1999 / 2500 / 2999` 的 ONNX 快照并完成配对快筛。
- [x] 确认后期 checkpoint 纯原地转向失败，且 `model_1999` 存在站立并忽略
  yaw 的局部最优现象。
- [x] 在行进转向协议下确认 `model_2500` 能双向转弯，但仍持续右偏、左右
  不对称且前进速度欠跟踪。
- [x] 完成 `model_2500` PointGoal pilot。
- [x] 通过 20-Episode 低速 yaw 归因确认左右启动死区来自底层 policy，
  并据此定义候选 B；不在同一 Run 中改配置，不同时
  修改多个因素。

后续 checkpoint 统一使用分层固定指令快筛，不再使用“直行 + 两条纯原地转向”作为
PointGoal 主要 Gate。

PointGoal 核心指令：

```text
[vx= 0.00, vy=0, wz= 0.00]  静止/停止
[vx=+0.10, vy=0, wz= 0.00]  低速前进
[vx=+0.30, vy=0, wz= 0.00]  正常直行
[vx=+0.25, vy=0, wz=+0.25]  缓慢左转
[vx=+0.25, vy=0, wz=-0.25]  缓慢右转
[vx=+0.20, vy=0, wz=+0.50]  正常左转
[vx=+0.20, vy=0, wz=-0.50]  正常右转
```

辅助 locomotion 指令：

```text
[vx=-0.30, vy= 0.00, wz= 0.00]  后退
[vx= 0.00, vy=+0.20, wz= 0.00]  左横移
[vx= 0.00, vy=-0.20, wz= 0.00]  右横移
[vx= 0.00, vy= 0.00, wz=+0.50]  原地左转
[vx= 0.00, vy= 0.00, wz=-0.50]  原地右转
```

另外独立执行“前进 `3～5 s` → 零指令停止”序列。七条核心指令、五条辅助指令和
停止序列分别汇总，不合并为一个成功率。行进转向统一采用“先建立步态，再叠加
yaw 指令”的协议；纯原地转向只作为辅助能力报告。

### 5.3 轻量 Classical PointGoal pilot

#### 目的与边界

使用 `model_2500.onnx` 和一个最小可用的 Constrained Go-to-Goal Controller，回答两个
问题：现有航向闭环能否抵消直行右偏，以及左右转向不对称是否会直接转化为
到达失败。本 pilot 只在 Nominal 物理参数下运行，不调参对比 Naive P 与受限控制器，
不替代第四周的 `200 Episode` 正式评测。

为了避免写出一次性脚本，pilot 仍遵循第四周的接口边界：

```text
MujocoBackend -> RobotState + GoalState -> Navigator -> VelocityCommand
```

`Navigator` 不得直接读取 `mj_data.qpos`；底层 ONNX 通过后端注入，以便新模型只需
替换路径便能按同一协议复测。

#### 实施顺序

- [x] 实现最小 `RobotState / GoalState / VelocityCommand / Navigator / MujocoBackend` 接口。
- [x] 实现固定参数的 Constrained Go-to-Goal：速度限幅、大航向误差时降低前进速度、
  接近目标时减速，并包含超时与跌倒终止。
- [x] 用不计入 pilot 的少量 smoke Episode 校验坐标系、目标判定、轨迹记录和指令限幅。
- [x] smoke 后一次性冻结 goal 坐标、success tolerance、timeout、controller 参数、
  reset 方式和 seeds；正式 pilot 中不再追着结果调参。
- [x] 运行 5 组对称目标：正前、左前、右前、左侧、右侧；左右使用镜像坐标和配对
  reset seed。
- [x] 每组目标 5 seeds，共 `5 × 5 = 25 Episode`；单独报告，不与行进转向的
  25 Episode 或第四周正式评测累加。
- [x] 记录 Success Rate、Final Distance、Path Efficiency、Completion Time、Fall Rate、
  Timeout Rate，以及左前/右前、左侧/右侧的镜像差异。
- [x] 附加记录 `vx/wz` 命令和实测值、限幅占比与轨迹，供失败归因使用。

#### pilot 结果的决策用法

pilot 是小样本诊断，不用它声称导航策略已验收通过。在运行前将下列判定项的
数值门槛写入配置或报告，运行后不追溯修改：总体与分类到达率、正前目标偏航、左右
成功率差、镜像轨迹差、超时和 Fall Rate。

| pilot 主要现象 | 优先归因 | 下一个单变量候选 |
|---|---|---|
| 正前也持续偏航，左右目标明显不对称，控制器长时间 yaw 限幅 | 底层 yaw 跟踪或奖励定义 | 优先只改 yaw tracking Reward 定义 |
| 方向基本正确，但持续低速造成超时和低路径效率 | 底层前进速度欠跟踪 | 只改 linear-velocity tracking 的一个因素 |
| 突发指令、大航向误差时失败，平稳弧线可成功 | 先排查高层指令形状和变化率 | 先修正 Navigator 约束；有证据后才训练底层 |
| 可到达且左右对称，但指令级评测仍有明显速度误差 | 导航闭环可补偿，但 locomotion 质量仍不足 | 以速度跟踪为唯一因素训练新候选 |

### 5.4 根据 pilot 证据训练新模型

不把“继续收紧 angular std”当成默认答案。PointGoal pilot 和随后的
20-Episode 低速 yaw 归因已确认主要失败是底层正 yaw 启动死区和左右不对称。
候选 B 只将 angular tracking
从“会受其他机身角速度影响的误差”改为“commanded yaw rate 与实测 yaw-axis
angular velocity 的误差”。保持 std、reward weight、command sampling、PPO 参数、
seed 和 envs 不变，不同时加 mirror loss。

新候选执行顺序：

- [x] 写明 pilot 与低速 yaw 归因证据、唯一训练假设、Control 和 Candidate 差异。
- [x] 用测试锁定只有一个预期配置或 Reward 定义发生改变。
- [x] 先运行 `64 envs × 5 iterations` CUDA 冒烟，再运行 `64 envs × 25 iterations`
  预运行；检查 NaN、跌倒、Reward 量级和行为退化。
- [x] 通过后使用 4096 envs 从头训练，在 `500 / 1000 / 1500 / 2000` 及必要的
  后续 checkpoint 保存与导出；不因为“还在上升”就无上限续训。
- [x] 对 `500 / 1000 / 1250 / 1500 / 1750 / 1999` 做行进转向快筛，与
  `model_2500` 使用配对 seeds；`model_1750` 在直行偏航和左右对称性上最佳。
- [x] 新候选在行进 yaw 对称性上明确改善且快筛 0 跌倒后，复测同一个
  25-Episode PointGoal pilot。
- [x] 对失败 checkpoint 补做立即/稳定启动、原地转向 checkpoint sweep、官方
  Reward Manager 和 ONNX 导出一致性检查，将退化定位到 1500–1750 的后期训练。
- [x] 对仍保留静止后双向启动能力的 `model_1500` 复测同一 pilot；结果为
  25/25 到达、0 跌倒、0 超时，左侧和右侧目标均为 5/5。
- [x] 先冻结 seed 42 的 `model_1500.onnx` 与 Controller，使用未参与 checkpoint
  选择的 evaluation seeds 100–104，运行 70-Episode Nominal holdout：距离
  `1.0/1.8 m`，方向 `0°/±30°/±60°/±90°`。运行前写死总体/逐目标成功率、
  Fall、Timeout 和镜像差门槛，运行中不调参。结果为 70/70 到达、0 跌倒、
  0 超时，14 个目标均 5/5，最大镜像成功率差为 0，全部预设 Gate 通过。
- [x] 保持 Candidate B 配置不变，使用 training seed 43 从头训练至
  `model_1500`。静止后双向启动均 5/5，行进转向 25 Episode 为 0 跌倒，同一
  PointGoal pilot 为 25/25、0 跌倒、0 超时；seed 42 的早停能力得到第二次复现。
- [x] seed 43 从 `model_1500` 续训至 `model_1999`，补测 `model_1750/1999`。
  `model_1750` 仍 25/25，但直行速度、完成时间和 Path Efficiency 已退化；到
  `model_1999`，稳定后右转仅 `wz=-0.019`。seed 42 与 43 均有 1500 后回退，
  但崩塌时点和方向不同，因此当前采用 `model_1500` 早停。
- [x] 在开始 OOD 鲁棒性测试前，固定 seed 42 的 `model_1500.onnx`，完成现有
  400-Episode 随机速度 Nominal 协议：reset seeds 42–441，100 standing、
  60 turn-in-place、240 三轴 general commands。400/400 未跌倒，但预先冻结的
  逐轴 RMSE Success 为 0%；完整随机速度能力不通过。Candidate B 的 yaw 均值
  slope/MAE 从 baseline 的 `0.801/0.146` 改善到 `1.040/0.078`，正负原地转向
  也接近对称，但 `vx` 仍欠跟踪、`vy` 能力弱且停止状态存在漂移和角速度振荡。
- [x] 延续已经确定的第一阶段任务边界：PointGoal Navigator 使用 `[vx,0,wz]`，
  非零 `vy` 记为范围外能力，不以完整三轴 Success 覆盖闭环到达结果；停止漂移、
  `vx` 欠跟踪和 yaw 振荡作为 OOD 必须持续观察的已知风险。
- [x] 完成 seed 42 `model_1500.onnx` 的 25-Episode PointGoal OOD 快筛。Nominal、
  friction 0.3、motor 80% 和 backlash 均 5/5、0 跌倒；delay 60 ms 仅 2/5，
  并有 3/5 跌倒，是当前最明确的任务层安全缺口。报告位于
  `results/week03/07_pointgoal_robustness/`。
- [x] 保持模型、Controller、目标和 seeds 不变，完成 delay 0/20/40/60 ms 配对评测：
  每档 25 Episode。0/20/40 ms 均 25/25、0 跌倒；60 ms 仅 8/25，并有 17/25
  跌倒。当前 20 ms 分辨率下，硬安全失效阈值位于 `(40,60] ms`。
- [x] 项目当前没有实机，因此不再等待真机延迟测量；完成冻结的 14 条件
  `350-Episode` 仿真鲁棒性矩阵。12/14 条件通过：全部 friction、motor proxy、
  backlash 及 delay 20/40 ms 均为 25/25、0 跌倒；delay 60 ms 为 8/25、17 次
  跌倒，delay 80 ms 为 0/25、25 次跌倒。快筛、延迟阈值和完整矩阵统一记录于
  `results/week03/07_pointgoal_robustness/`。
- [x] 根据 Nominal、OOD 和 PointGoal 两层结果，将 seed 42 `model_1500.onnx`
  冻结为后续导航阶段的仿真下层基线，并明确记录 `(40,60] ms` 延迟失效边界；
  这不等于 Sim2Real-ready。
- [ ] 进入 Week 4 Classical PointGoal 正式 benchmark。若以后单独开展下层鲁棒性
  增强，则只加入 action-delay randomization，并复跑同一 Nominal 与 350-Episode
  矩阵，不同时修改 yaw-only Reward、command sampling 或多个 curriculum 权重。

候选 B 的训练前快照：官方仓库分支 `codex/yaw-only-angular-tracking`，Task 为
`Mjlab-Velocity-Flat-Yaw-Only-Tracking-MicroDuck`；`std=0.5`、reward weight、
command sampling、PPO、seed 42 和从头训练设置均继承候选 A。数值单测与配置差异
测试共 5 项通过，`64×5` 和 `64×25` 两级 CUDA 预运行均正常结束且
`nan_state=0`。短预运行从随机策略开始，跌倒率与 Reward 只用于检查运行异常，
不得作为模型质量或候选优劣结论。正式训练已在 commit `062921c` 上使用
4096 envs、seed 42 从头完成至 `model_1999.pt`；训练曲线显示 yaw error 改善，
但末段跌倒统计未同步改善。ONNX 行进转向快筛共 150 Episode、0 跌倒，选出
`model_1750`；它的正常行进转向已接近镜像对称，但 `vx=0.05,wz=+0.50` 的
低速正 yaw 响应仅 4.5%。同协议 PointGoal pilot 中，`model_1750` 与
`model_2500` 均为 20/25 到达、0 跌倒；前者左侧目标仍 0/5，且中位
Final Distance 从 0.935 m 退化到 1.498 m，其余目标的完成时间也更长。
进一步的 10-Episode 原地转向诊断显示，在 `vx=0,wz=±0.5` 下实测 yaw
响应只有 3.1%～3.9%，左右都几乎站立不动。虽然训练采样器已有 15%
左右对称的原地转向桶，策略仍未学会该能力；因此低速联合命令覆盖不足
不是唯一根因。后续 checkpoint 诊断进一步发现：`model_500/1000/1250/1500`
在 1 秒零命令稳定后均能以 `wz=±0.5` 完成 5/5 双向转向，而
`model_1750/1999` 两侧均变为 0/5，能力在 1500–1750 间发生突变。
`model_1750` 在 reset 后立即下命令时仍有 3/5～5/5 的转向成功，但先站立
1 秒后只剩低速负 yaw 为 5/5，其余三项为 0/5，说明后期策略形成了强烈的
状态历史依赖型站立吸引域。

官方 PT Reward 归因没有发现纯 yaw 关闭 gait Reward 的实现错误；静止失败
主要损失 angular tracking 与 air time，总 Return 也低于正确转向。重新导出的
ONNX 与现有模型在 100 组随机 Observation 上 Action 完全一致，可排除错误
checkpoint 或导出不一致。由于训练在 1500 附近同时调整 action-rate、standing
fraction、head-pose 和 CoM 等多项 curriculum，本轮尚不能把退化单独归因给
某一个权重。随后使用完全相同的冻结 Controller 对 `model_1500` 运行
25-Episode PointGoal pilot：五类目标全部 5/5 到达，合计 25/25、0 跌倒、
0 超时；相同协议下 `model_2500` 和 `model_1750` 均为 20/25。说明 yaw-only
Reward 在合适 checkpoint 上已经产生任务层改善，但路径效率和完成时间仍不及
Candidate A 已成功的目标；这一步完成时证据还只来自 seed 42。
使用未见 evaluation seeds 和扩展目标完成的 70-Episode holdout 为 70/70、
0 跌倒、0 超时，确认了固定 seed 42 `model_1500.onnx` 的 Nominal 能力。随后
training seed 43 的 `model_1500` 在同一 pilot 中也达到 25/25、0 跌倒、0 超时，
说明早停能力已在两个独立训练中复现；但 seed 43 的路径效率略低，直行偏航方向也与
seed 42 相反。两个 seed 继续训练后都出现性能回退：seed 42 在 1750 双侧启动崩塌，
seed 43 先整体变慢并在 1999 出现右侧启动崩塌。训练方法的早期有效性与后期不稳定性
均有了跨 seed 证据，`model_1500` 因而升级为推荐早停 checkpoint；它仍需先完成
400-Episode 随机速度检测。检测中 400/400 未跌倒，但严格三轴逐步 RMSE Success
为 0%；yaw 的 Episode 均值响应和左右对称性相对 baseline 明显改善，`vx` 欠跟踪、
非零 `vy`、停止漂移和步态内 yaw 振荡仍是明确能力边界。该随机协议比当前
`[vx,0,wz]` PointGoal 接口更宽，因此结果不覆盖 70/70 闭环 holdout，也不能被
闭环到达率掩盖。第一阶段继续维持 `vy=0` 的既定范围。随后完成的 PointGoal OOD
快筛显示，friction 0.3、motor 80% 和 backlash 未立即退化，但 delay 60 ms 导致
3/5 跌倒，并同步放大停止漂移、`vx/wz` tracking error 和 yaw 振荡。下一步先用
20/40 ms 配对评测定位延迟阈值，再决定是否设计 action-delay randomization
单因素新候选。

### 5.5 必做鲁棒性参数

完整鲁棒性矩阵放在新模型训练之后：候选必须先通过 Nominal 固定指令快筛，并在
同一 25-Episode PointGoal pilot 中相对 `model_2500` 显示出目标改善。当前不单独为
已知持续右偏和速度欠跟踪的 `model_2500` 消耗完整矩阵预算。

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

### 5.6 控制器级鲁棒性评测

- [x] 为四类必做不确定因素建立统一配置入口：采用同一冻结 ONNX、Controller、
  对称目标与配对 seeds，每次只注入一个因素；支持协议签名、逐 Episode 进度和
  断点续跑。四种注入均已完成不计入正式结果的短 smoke。
- [ ] 完成新旧候选的 Nominal 固定指令和 PointGoal pilot 配对对比。
- [x] 对当前冻结的 seed 42 `model_1500.onnx` 运行 25-Episode PointGoal OOD
  快筛：Nominal、friction 0.3、delay 60 ms、motor 80% 和 backlash 各覆盖五类
  对称目标；结果为 22/25 到达，所有 3 次跌倒均来自 delay 60 ms。
- [x] 完成 delay 阈值定位：40 ms 仍通过完整任务/安全 Gate，60 ms 在所有五类目标
  上均出现跌倒或成功率下降；不直接运行完整四因素矩阵。
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

- [x] 官方 `alpha_walking.onnx` 独立评测报告
- [x] angular tracking std 单变量训练报告
- [x] `model_2500` PointGoal 行进转向报告
- [x] `model_2500` 的 25-Episode Classical PointGoal pilot 报告与轨迹
- [x] `model_2500` 的 20-Episode 低速 yaw 归因报告与图表
- [x] 由 pilot 证据驱动的新模型单变量训练报告
- [x] 新旧候选在同一 Locomotion 协议与 PointGoal pilot 下的配对对比
- [ ] `model_5999` / 官方模型 / 自训练候选的 Nominal 对比表
- [x] 四类不确定因素的统一配置入口、短 smoke 与断点续跑框架
- [x] 25-Episode OOD 快筛、关键汇总与敏感性图
- [ ] Delay 阈值定位和完整四因素实验数据
- [ ] 最优两个模型的鲁棒性汇总表
- [ ] 2～4 张敏感性或对比曲线
- [ ] Reality Gap / Sim2Real proxy 影响分析
- [ ] 典型 Failure Cases
- [ ] 一个冻结供第四周使用的 Locomotion Policy

### 验收标准

最终导航底层候选至少满足：

- 行进中正负转向方向均正确，且不是单个 checkpoint 的偶然行为；
- 直行偏航显著低于 `model_5999`；
- 低速命令存在可预测响应，静止保持稳定；
- Nominal 条件无明显跌倒；
- OOD 退化能够量化，且不是轻微扰动下立即全面失效。

冻结前至少完成：

```text
官方模型评测
    +
自训练单变量候选
    +
25-Episode PointGoal pilot
    +
根据 pilot 训练并复测新候选
    +
4 类 Uncertainty
    +
最优两个底层模型统一对比
    +
冻结第四周底层策略
```

若某个模型通过上述门禁，则冻结其 ONNX、Normalizer、上游 commit、评测配置和
SHA-256 后进入第四周。若所有模型均未通过，第四周可继续完善接口与控制器
单元测试，但不得把使用不合格底层得到的 pilot 结果写成最终导航结论。

> 完成本周后结束 M1 的主要实验，不继续无限调步态，立即转入 M2。

---

## 6. 第 4 周：PointGoal Navigation 与传统控制对比

### 本周目标

使用第三周在轻量 pilot、新旧候选复测和 Nominal/OOD 对比后冻结的底层策略，
建立随机 Start/Goal 条件下的自主目标点导航，并形成
一个可由第五周 RL Navigator 直接复用的 Classical Navigation Benchmark。
第三周的 25-Episode pilot 只用于诊断和底层选型；本周重新冻结正式 controller
参数和随机 Start/Goal 协议，不继承 pilot 的 Episode 计数。

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

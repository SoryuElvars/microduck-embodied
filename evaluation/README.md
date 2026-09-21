# Locomotion Benchmark

本目录用于实现 MicroDuck Locomotion 的自动、可复现评测。评测代码属于
`microduck-embodied`，但继续复用官方 `microduck_rl` 的 MuJoCo 模型、BAM M6
执行器、Observation 构造和 ONNX 推理组件。

## 当前阶段：8 项指令响应矩阵

当前入口已经能够：

- 解析并校验版本化的速度命令集；
- 校验官方仓库、场景文件和 ONNX 是否存在；
- 固定随机种子和结果目录；
- 打印预期的 `61 obs -> 14 actions` 合约及 Episode 计划。
- 复用官方 MuJoCo、BAM M6、Observation 和 ONNX Policy；
- 无窗口执行 1 秒零命令预热和 10 秒固定命令；
- 将 61 维 Observation、14 维 Action、速度、位置和姿态写入逐步 CSV；
- 输出速度 RMSE、平均速度、净偏航、位移和跌倒代理指标。
- 以相同的 20 个种子运行 8 项固定指令，共 160 个 Episode；
- 生成每项指令的独立 summary 和跨指令 suite summary。

## 400-Episode 随机速度基准

固定八指令矩阵用于单轴诊断；随机速度基准另用 400 个 Episode 覆盖组合指令，
两者分别报告，不合并成功率。随机协议使用 reset seeds `42–441` 和 command seed
`20260911`，显式分配 25% 静止、15% 原地转向和 60% 全范围均匀采样：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/random_locomotion_benchmark.py \
  --policy logs/rsl_rl/velocity/2026-09-05_22-08-33_baseline-flat-resume-5000/microduck_velocity_flat_5999.onnx \
  --episodes 400 \
  --seed 42 \
  --command-seed 20260911 \
  --output-dir ~/projects/microduck-embodied/artifacts/week02/05_random_400/run
```

运行器会逐 Episode 保存进度，同一协议和输出目录可以断点续跑。处理结果：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/analyze_random_locomotion.py \
  --input ~/projects/microduck-embodied/artifacts/week02/05_random_400/run/summary/random_velocity_400_microduck_velocity_flat_5999.json \
  --output-json ~/projects/microduck-embodied/results/week02/05_random_400/summaries/random_velocity_400_processed.json \
  --output-figure ~/projects/microduck-embodied/results/week02/05_random_400/figures/random_velocity_400_response.png \
  --output-report ~/projects/microduck-embodied/results/week02/05_random_400/README.md
```

逐步 CSV、断点文件和源汇总位于 `artifacts/week02/05_random_400/run/`，只保留
在本地；处理后的关键汇总、图表和报告位于 `results/week02/05_random_400/`。
Nominal Success Rate 是本项目定义的指标，不是上游官方指标。

该部署形态评测不加载训练环境的 Reward Manager，因此汇总文件明确记录
`reward_available: false`。Episode Return 由下面独立的官方环境评测记录，不会从
ONNX 部署轨迹中伪造 Reward。

## Episode Return（官方 Reward Manager）

Episode Return 评测从官方 `microduck_rl` 环境加载 `model_5999.pt`，
恢复 checkpoint 保存的课程进度和最终奖励权重，然后在 8 项固定指令下分别
运行 5 个可复现、成对的向量环境初始状态。每个 Episode 为 10 秒，总计
40 个 Episode。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/episode_return_benchmark.py
```

评测器在每个 50 Hz 控制步记录 `reward_buf`，Episode Return 定义为
`sum(reward_buf)`，并同时求和 16 个已加权奖励项。每一步都会检查奖励项之和
与 `reward_buf` 一致。名义配置关闭观测噪声、推力与域随机化，但保留官方
reset、终止条件、BAM 执行器、Reward Manager 和 checkpoint 的课程状态。

评测器会读取 command set 的 `warmup_s`：预热期间运行 policy 但固定零命令，
预热 Reward 不计入测试段 Return，随后才切换到目标命令。可以用
`--warmup-s 0` 显式构造 reset 后立即下命令的对照。输出除 Return 和各 Reward
项外，还记录测试段的 `mean_actual_vx/vy/wz`，用于判断高 Reward 是否真的对应
目标运动。旧结果文件会保留其当时记录的 `protocol.warmup_s`，不得与新协议
静默合并。

结果：
`results/week02/01_baseline_5999/summaries/episode_return_8x5_model_5999.json`。
这组 40 Episode 是独立协议，不与 ONNX 固定指令的 160 Episode 或随机指令的
400 Episode 合并计数。

## Week 03 自训练候选 Checkpoint Gate

Angular std `sqrt(0.25)` 候选在 `500 / 1000 / 1500 / 2000` checkpoints
分别导出 ONNX，并使用独立的三指令配对快筛：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy <EXPORTED_CHECKPOINT_ONNX> \
  --command-set ~/projects/microduck-embodied/evaluation/command_sets/angular_std025_gate_5.json \
  --output-dir ~/projects/microduck-embodied/artifacts/week03/02_angular_std025_candidate/<CHECKPOINT>
```

该 Gate 每个 checkpoint 独立计算 3 条指令 x 5 seeds，不与第二周的固定
160 Episode、随机 400 Episode 或官方 Return 协议合并成功率。

## PointGoal 行进转向门禁

PointGoal 的主要下层能力不是从静止状态原地旋转，而是直行时保持较低偏航，
并在已经建立步态后响应同时非零的 `vx + wz` 指令。该独立协议先用 1 秒零命令
稳定仿真，再以 `vx=0.25 m/s` 直行 2 秒建立步态，最后测量 8 秒直行、缓转或
正常转弯；预备段不计入测试段指标。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy ~/projects/microduck-embodied/artifacts/week03/02_angular_std025_candidate/models/model_2500.onnx \
  --command-set ~/projects/microduck-embodied/evaluation/command_sets/pointgoal_moving_turn_5.json \
  --output-dir ~/projects/microduck-embodied/artifacts/week03/03_pointgoal_moving_turn/model_2500
```

协议使用直行 `(0.25, 0)`、缓转 `(0.25, ±0.25)` 和正常转弯
`(0.20, ±0.50)`，每项运行配对 seeds `42--46`，共 25 Episode。它与原地
转向 Gate、随机组合指令和官方 Return 均分别报告。

## Classical PointGoal pilot 框架

pilot 使用 `model_2500.onnx` 与最小 Constrained Go-to-Goal Controller，检查
航向闭环能否补偿底层持续偏航。框架分为：

- `navigation/types.py`：`RobotState / GoalState / VelocityCommand / Navigator`；
- `navigation/classical_navigator.py`：限速、限加速度且大航向误差时保留小幅前进的
  Go-to-Goal Controller；
- `navigation/mujoco_backend.py`：复用官方 MuJoCo、BAM M6、Observation 和 ONNX
  推理，并将仿真真值封装为 `RobotState`；
- `evaluation/pointgoal_pilot.py`：Episode 循环、终止判定、轨迹记录与左右镜像汇总；
- `evaluation/goal_sets/pointgoal_pilot_5.json`：5 组对称目标和已冻结的 pilot 协议参数。

Navigator 只消费 `RobotState`，不直接读取 `mj_data.qpos`。当前位姿来源在汇总中明确记为
`simulator_ground_truth_via_backend`；它是部署形态的 Nominal pilot，不是真机定位或
Reality Gap 结论。

首先只校验文件、模型和 `61 obs -> 14 actions` 合约：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_pilot.py \
  --validate-only
```

运行正式 pilot 前，可以用不计入正式结果的短程 smoke 复核执行链路：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_pilot.py \
  --smoke \
  --smoke-goal front
```

`--smoke-goal` 可选 `front / front_left / front_right / left / right`。当前 smoke 已完成，
goal 坐标、成功半径、超时、controller 参数、reset 方式和 seeds 已一次性冻结，
`protocol_status` 为 `frozen`。正式 25 Episode 期间不得根据结果追调这些参数。

原始 50 Hz CSV 和 smoke summary 写入
`artifacts/week03/04_classical_pointgoal_pilot/model_2500/`，不提交 Git。ONNX pilot 不加载
Reward Manager，因此 `reward_available` 为 `false`。失败 Episode 不伪造 Path Efficiency，
另外记录取值在 `[0, 1]` 的 Progress Efficiency 供诊断使用。

将多个 smoke 或正式 Episode 统一变换到各自的初始机体坐标系后绘图：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/plot_pointgoal_pilot.py \
  --summary \
    ~/projects/microduck-embodied/artifacts/week03/04_classical_pointgoal_pilot/model_2500/summary/pointgoal_smoke_front_left_model_2500.json \
    ~/projects/microduck-embodied/artifacts/week03/04_classical_pointgoal_pilot/model_2500/summary/pointgoal_smoke_front_right_model_2500.json \
  --output \
    ~/projects/microduck-embodied/artifacts/week03/04_classical_pointgoal_pilot/model_2500/figures/smoke_front_pair.png
```

使用 MuJoCo 原生 viewer 实时观看一个不计入正式 pilot 的 Episode：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_pilot.py \
  --view \
  --view-goal left \
  --view-seed 42
```

viewer 中橙色球是目标点，半透明绿色圆盘是 `0.2 m` 成功区域。窗口使用
跟随 Duck 的相机；Episode 结束后窗口会保留，关闭 MuJoCo 窗口即退出程序。
`--view-goal` 支持 `front / front_left / front_right / left / right`，
`--view-seed` 可选择具体 reset seed。观看模式按实时速度运行，其轨迹单独写入
`artifacts/week03/04_classical_pointgoal_pilot/model_2500/viewer/`，不覆盖正式结果。

如果 WSL 无法弹出窗口，先在 WSL 中检查 `echo $DISPLAY` 是否有值；Windows 11 + WSLg
通常无需额外 X Server。

## Week 4 Classical PointGoal Benchmark v1

第四周使用新的正式入口 `evaluation/pointgoal_benchmark.py`，不会读取或覆盖第三周 pilot
结果。任务是位置型 PointGoal：输入等价于
`[delta_x_body, delta_y_body] / [distance, heading_error]`，输出固定为
`[vx, 0, wz]`；`heading_error` 表示当前朝向到目标点的误差，不要求最终目标 yaw。
指定最终姿态属于以后单独扩展的 PoseGoal。

版本化协议位于 `evaluation/pointgoal_protocols/week04_classical_v1.json`：

- 200 个 `EpisodeSpec` 已由固定生成 seed 生成并保存，分别记录 reset seed、世界系
  Start/Goal、timeout、成功半径、保持时间和 2 秒 post-arrival 观察时间；
- Episode 清单状态为 `frozen`，并带独立 SHA-256；运行时不再随机采样场景；
- Naive P 与 Constrained 参数已在 smoke/quick 通过后确认，协议整体状态为 `frozen`；
  后续正式评测不得根据结果追调 Controller 参数；
- 两个 Controller 共享 `0.25 m/s` 最大前进速度、`0.5 rad/s` 最大 yaw rate、
  `0.15 m` Goal tolerance、同一个 `model_1500.onnx` 和完全相同的 Episode 顺序；
  任务成功半径独立保持为 `0.20 m`，提供停止余量。

检查生成器仍能精确复现冻结的 Episode 清单、选择和清单哈希：

```bash
cd ~/projects/microduck-embodied
python3 evaluation/generate_pointgoal_protocol.py --check
```

静态校验协议、模型 SHA、官方 commit 与 ONNX 文件：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_benchmark.py \
  --validate-only
```

headless smoke 使用冻结清单中最接近正前、左、右、后方的 4 个场景，对两个 Controller
各运行一次，共 8 个 Controller-Episode。smoke 的导航时限缩短为 5 秒，只验证加载、
reset/goal 注入、终止、post-arrival、记录和输出，不计入 quick 或 formal：

```bash
uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_benchmark.py \
  --smoke
```

Viewer 每次只观看一个 Controller/Episode；当前四个冻结 smoke ID 为
`front=w04e008 / left=w04e017 / right=w04e080 / behind=w04e072`：

```bash
uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_benchmark.py \
  --viewer \
  --controller naive_p \
  --episode-id w04e008
```

快筛固定使用清单中的 10 个共享随机场景，对两个 Controller 各运行一次，共 20 个
Controller-Episode：

```bash
uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_benchmark.py \
  --quick
```

runner 每个 Episode 写入带 run signature 的进度文件并默认断点续跑。协议、场景清单、
模型、官方 commit、评测代码或运行模式改变后，旧进度会被拒绝。smoke、quick 和以后
人工冻结后的 full 分别写入
`artifacts/week04/01_classical_pointgoal_benchmark/model_1500/<mode>/`，不会混合计数。

汇总分别报告 Success、Final Distance、到达前 Path Length/Efficiency、Completion Time、
Fall/Timeout/Invalid State、`vx/wz` 饱和比例、post-arrival 停止漂移和重新离开成功半径，
并输出逐场景 `Constrained - Naive P` 配对差值。ONNX 运行仍没有训练 Reward。

2026-09-21 只把共享 Goal tolerance 从 `0.20 m` 改为 `0.15 m` 后，使用同一组 10 个
EpisodeSpec 重跑 quick：两个 Controller 都是 10/10 到达、0 fall、0 timeout、
0 invalid state，post-arrival re-departure rate 均从 100% 降至 0%。两个 Controller
及协议随后冻结，尚未运行 `--full`。阶段表格、单变量对照、配对差值和决策边界见
`results/week04/01_classical_pointgoal_benchmark/README.md`。

## PointGoal Nominal holdout

当某个 checkpoint 通过 25-Episode pilot 后，先验证这个固定 ONNX 的能力，再决定
是否投入多个 training seeds。独立 holdout 使用
`evaluation/goal_sets/pointgoal_nominal_holdout_70.json`，在运行前冻结：

- `model_1500.onnx` 和既有 Classical Controller；
- 未用于 checkpoint 选择的 reset seeds `100–104`；
- 两个距离 `1.0 / 1.8 m`；
- 七个对称方向 `0° / ±30° / ±60° / ±90°`；
- 14 个目标 × 5 seeds，共 70 Episode；
- 1 秒零命令预热、20 秒 timeout、0.2 m 到达半径和 0.5 秒保持时间。

运行顺序固定为：协议与 I/O 校验 → 一次性运行 70 Episode → 总体、逐目标、
距离、角度和镜像对比 → 通过后才进入 OOD 或 training-seed 复现。运行中不得调整
Controller、timeout 或目标。

验收门槛同样在运行前写入 goal set：总体 Success Rate ≥95%，每个目标 ≥80%，
Fall Rate=0，Timeout Rate ≤5%，任一镜像组左右成功率差绝对值 ≤20 个百分点。
Path Efficiency 与 Completion Time 必须报告，但本轮不以运行后观察到的数值追设
门槛。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_pilot.py \
  --policy ~/projects/microduck-embodied/artifacts/week03/06_yaw_only_tracking_candidate/models/model_1500.onnx \
  --goal-set ~/projects/microduck-embodied/evaluation/goal_sets/pointgoal_nominal_holdout_70.json \
  --output-dir ~/projects/microduck-embodied/artifacts/week03/06_yaw_only_tracking_candidate/pointgoal_nominal_holdout/model_1500
```

该协议仍是 ONNX + 原生 MuJoCo + BAM M6 的 Nominal 仿真，不加载训练 Reward，
也不能替代 Reality Gap 或真机测试。

`model_1500.onnx` 已按该冻结协议完成 70 Episode：70/70 到达、0 跌倒、0 超时，
14 个目标均为 5/5，全部预设 Gate 通过。关键汇总与图表位于
`results/week03/06_yaw_only_tracking_candidate/`；原始逐步 CSV 仅保存在
`artifacts/week03/06_yaw_only_tracking_candidate/pointgoal_nominal_holdout/`。

## PointGoal OOD 鲁棒性框架

鲁棒性评测冻结同一个 `model_1500.onnx`、同一个 Constrained Go-to-Goal
Controller、同一组目标与配对 reset seeds，每次只修改一个部署侧因素。统一入口为
`evaluation/pointgoal_robustness.py`，版本化矩阵位于
`evaluation/robustness_configs/pointgoal_ood_v1.json`：

- Ground friction：`0.3 / 0.5 / 0.7 / 0.9 / 1.1`；
- Actuator delay：`20 / 40 / 60 / 80 ms`，复用官方 PolicyInference action buffer；
- Motor strength proxy：BAM 供电电压的 `80% / 90% / 110%`，`100%` 由 Nominal
  条件提供；它是电压代理量，不等同于宣称所有电机 torque 被严格线性缩放；
- Backlash：官方 `scene_walk_backlash.xml` 的 `2 deg` 总齿隙模型。策略仍观察
  actuated motor-side joint state；汇总会显式记录这项限制。

先只校验矩阵、模型和目标协议：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_robustness.py \
  --validate-only
```

四类注入均已用 `front / seed 42 / 5 s` 的非正式 smoke 验证执行链路。需要单独
复核时使用 `--smoke-condition`，其结果不计入快筛：

```bash
uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_robustness.py \
  --smoke \
  --smoke-condition backlash_2deg \
  --smoke-goal front
```

第一层快筛固定为 `Nominal + friction 0.3 + delay 60 ms + motor 80% + backlash`，
5 个条件 × 5 个对称目标 × 1 个配对 seed，共 25 Episode：

```bash
uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_robustness.py \
  --quick
```

快筛只把 Fall Rate 和 invalid-state rate 作为安全门槛；到达率、Timeout、最终距离、
Path Efficiency、左右镜像差、`vx/wz` 跟踪 RMSE、零命令速度/角速度和预热段位姿
漂移都报告，但不把每条件仅 5 Episode 的结果包装成统计性结论。完整矩阵用 14 个
不重复条件 × 25 Episode，共 350 Episode；必须在快筛检查后另行运行，不由 smoke
自动触发：

```bash
uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_robustness.py \
  --full
```

运行器逐 Episode 写入带协议签名的进度文件，默认可断点续跑；模型、官方 commit、
评测代码、鲁棒性配置、目标集或运行模式变化时会拒绝混用旧进度。原始轨迹和进度只写入
`artifacts/week03/08_pointgoal_robustness/`。得到 quick 或 full suite summary 后生成
四因素敏感性图：

```bash
uv run python \
  ~/projects/microduck-embodied/evaluation/plot_pointgoal_robustness.py \
  --summary <POINTGOAL_ROBUSTNESS_SUMMARY.json> \
  --output <OUTPUT.png>
```

Nominal、quick OOD 和 full OOD 各自报告，不与 400-Episode 随机速度基准合并
Success Rate。当前 Navigator 范围仍为 `[vx,0,wz]`；`vy` 不纳入本轮 OOD Gate。

2026-09-18 的 `model_1500` quick screen 已完成：Nominal、friction 0.3、motor 80%
和 backlash 均为 5/5、0 跌倒；delay 60 ms 为 2/5，并发生 3/5 跌倒。随后以每档
25 Episode 补测 0/20/40/60 ms，并最终完成冻结的 14 条件 × 25 Episode 完整矩阵。
12/14 条件通过：全部 friction、motor proxy、backlash 及 delay 20/40 ms 均为
25/25、0 跌倒；delay 60 ms 为 8/25、17 次跌倒，delay 80 ms 为 0/25、25 次
跌倒。当前离散控制周期下，硬安全失效阈值位于 `(40,60] ms`。快筛分析见
`results/week03/07_pointgoal_robustness/README.md`；延迟阈值和完整矩阵已经合并
到同一份 Week03/07 报告中。

### Mass 与低层传感器噪声扩展

第二层扩展使用独立的版本化配置
`evaluation/robustness_configs/pointgoal_sensor_mass_v1.json`，避免修改已经完成并
用于350 Episode核心矩阵的冻结配置。扩展仍固定 seed 42 `model_1500.onnx`、同一
Constrained Go-to-Goal Controller、五个对称目标和配对 reset seeds，每次只改变
一个因素：

- `trunk_base` mass 与 diagonal inertia 同比缩放至 `90% / 110%`；
- IMU Low：`base_ang_vel ±0.03 rad/s`、`projected_gravity ±0.01`；High 为2倍；
- Joint Encoder Low：`joint_pos ±0.001 rad`、`joint_vel ±0.25 rad/s`；High 为2倍。

Low 档来自官方训练配置 commit `062921c`；High 档是明确标记的2倍仿真压力测试，
不是实机测量。噪声按 Episode seed 确定，在每个50 Hz控制步独立均匀采样，只修改
低层 actor observation。Navigator 位姿和 GoalState 继续使用仿真真值；本协议也不
包含常量 IMU 安装偏差或 encoder bias。

配置已在三类注入 smoke 后冻结。静态校验命令为：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_robustness.py \
  --config ~/projects/microduck-embodied/evaluation/robustness_configs/pointgoal_sensor_mass_v1.json \
  --validate-only
```

2026-09-19 使用 `front / seed 42 / 5 s` 完成不计入正式结果的代表性 smoke：

- Mass 90%：`trunk_base` mass 从 `0.199224` 变为 `0.1793016`，三个 inertia
  分量的比例也均为 `0.9`；
- IMU High：首步只改变 Observation `0:6`，最大绝对差为 `0.050914`，未超过
  `0.06` 的配置边界；
- Encoder High：首步 Observation `0:6` 不变，joint position/velocity 最大差分别为
  `0.001914 rad` 和 `0.480545 rad/s`，未超过 `0.002/0.5` 的配置边界。

三个 smoke 均无跌倒和 invalid state，但都在5秒 smoke 时限到达前终止；这只用于
验证注入链路，不作为行为 Gate。冻结后的 quick 为6个条件 × 5个目标 × 1个配对 seed，
共30 Episode。原计划只对出现安全、到达、tracking或漂移退化的条件选择性扩展；
为用统一的五 seeds 证据关闭第三周范围，实际将冻结的六个条件都扩展到每条件
25 Episode，且没有在看到结果后改变档位。

30-Episode quick 已完成：六个条件均为 `5/5` 到达、0跌倒、0超时、0 invalid state，
左右镜像成功率差均为0，全部 quick safety Gate 通过。High档没有暴露任务级失败；
相对 Nominal，较明显但仍属单 seed 描述性的变化是 IMU High 的零命令 yaw RMS
增加15.2%，以及 Encoder High 的预热平面漂移增加15.5%。原始结果使用独立目录：

```text
artifacts/week03/08_pointgoal_robustness/model_1500/sensor_mass_v1/
```

正式扩展矩阵已于2026-09-19完成：每条件25 Episode，共150 Episode。六个条件均为
25/25到达，合计0跌倒、0超时、0 invalid state，每目标到达率100%，左右镜像成功率
差为0，六个 Full Gate 全部通过。相对同一组 seeds/目标的25-Episode Nominal，Path
Efficiency 最大绝对变化1.0%，完成时间最大变化2.3%；High噪声的预热 yaw 漂移相对
增幅较大，但绝对量不超过0.033 rad，且未伴随任务级失败。处理后汇总位于：

```text
results/week03/07_pointgoal_robustness/summaries/pointgoal_sensor_mass_full_model_1500_processed.json
```

以下为可中断续跑的复现命令：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/pointgoal_robustness.py \
  --config ~/projects/microduck-embodied/evaluation/robustness_configs/pointgoal_sensor_mass_v1.json \
  --full
```

## PointGoal 低速 yaw 归因

用固定低速命令移除 Navigator 闭环，检查侧向目标的左右启动差异是否来自
底层 policy：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy ~/projects/microduck-embodied/artifacts/week03/02_angular_std025_candidate/models/model_2500.onnx \
  --command-set ~/projects/microduck-embodied/evaluation/command_sets/pointgoal_low_speed_yaw_5.json \
  --output-dir ~/projects/microduck-embodied/artifacts/week03/05_pointgoal_low_speed_yaw/model_2500 \
  --seed 42
```

协议测试 `vx=0.05/0.10` 与 `wz=±0.50` 的四种组合，每条五个配对 seeds，
共 20 Episode。它不使用 gait lead-in，不加载 Reward Manager，也不与
PointGoal pilot 或行进转向协议合并。

如需进一步区分低速联合命令覆盖和静止启动问题，使用配对的
10-Episode 原地转向辅助诊断：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy ~/projects/microduck-embodied/artifacts/week03/06_yaw_only_tracking_candidate/models/model_1750.onnx \
  --command-set ~/projects/microduck-embodied/evaluation/command_sets/pointgoal_turn_in_place_5.json \
  --output-dir ~/projects/microduck-embodied/artifacts/week03/06_yaw_only_tracking_candidate/turn_in_place_diagnostic/model_1750 \
  --seed 42
```

它只测试 `vx=0, vy=0, wz=±0.50`，每侧 5 个配对 seeds，仍属于辅助诊断，
不作为 PointGoal 主验收门槛。

若要区分 reset 初始状态和“先稳定站立再转向”，使用完全相同的四条命令运行
立即启动对照：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy ~/projects/microduck-embodied/artifacts/week03/06_yaw_only_tracking_candidate/models/model_1750.onnx \
  --command-set ~/projects/microduck-embodied/evaluation/command_sets/pointgoal_turn_startup_immediate_5.json \
  --output-dir ~/projects/microduck-embodied/artifacts/week03/06_yaw_only_tracking_candidate/turn_startup_immediate/model_1750 \
  --seed 42
```

该命令集以 `warmup_s=0` 测试 `vx=0/0.05,wz=±0.50`。它必须和 1 秒零命令
预热结果分开报告；两者差异衡量的是启动状态历史依赖，不是普通的重复测量。

官方 Reward 归因使用 PT checkpoint；默认从命令集继承 1 秒零命令预热：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/episode_return_benchmark.py \
  --task Mjlab-Velocity-Flat-Yaw-Only-Tracking-MicroDuck \
  --checkpoint <CANDIDATE_B_PT_CHECKPOINT> \
  --command-set ~/projects/microduck-embodied/evaluation/command_sets/pointgoal_reward_attribution_5.json \
  --output <OUTPUT_JSON>
```

如需立即启动对照，在同一命令后增加 `--warmup-s 0`。ONNX 部署轨迹仍不计算
训练 Reward；这里的 Return 只来自官方 PT checkpoint + Reward Manager。

## 汇总 TensorBoard 训练曲线

用实际训练谱系合并初训与续训 event，按 checkpoint 前 100 iterations 汇总，
并输出原始长表 CSV、关键 JSON 和对比图：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/summarize_training_metrics.py \
  --series candidate_a=logs/rsl_rl/velocity_angular_std025/2026-09-14_20-50-41_angular-std025-seed42-from-scratch,logs/rsl_rl/velocity_angular_std025/2026-09-14_22-04-10_angular-std025-seed42-resume-750-to-2000 \
  --series candidate_b=logs/rsl_rl/velocity_yaw_only_tracking/2026-09-16_09-56-12_yaw-only-tracking-seed42-from-scratch \
  --output-csv ~/projects/microduck-embodied/artifacts/week03/06_yaw_only_tracking_candidate/training_metrics.csv \
  --output-summary ~/projects/microduck-embodied/results/week03/06_yaw_only_tracking_candidate/summaries/training_metrics.json \
  --output-figure ~/projects/microduck-embodied/results/week03/06_yaw_only_tracking_candidate/figures/candidate_a_vs_b_training.png
```

同一 `series` 中后列出的续训 event 会覆盖重复 iteration，以免 checkpoint 恢复处
重复计数。`track_angular_velocity` 和总 Reward 在候选间定义不同，只能用于检查
各自训练趋势；跨候选判断优先使用定义未变的 `error_vel_yaw`、`error_vel_xy`、
episode length、termination 和后续部署评测指标。

## 绘制单 Episode 诊断图

使用 `matplotlib` 将原始 CSV 转换为四联图：`vx`、`vy`、`wz` 的 Target vs
Actual，以及对齐初始朝向的机体坐标系 XY 轨迹。浅色细线表示 50 Hz 原始数据，
深色粗线表示默认 1 秒滑动平均，虚线表示目标命令或积分得到的目标轨迹。
可使用 `--title` 为站立成功、典型失败等不同样本指定标题。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/plot_locomotion_episode.py \
  --input ~/projects/microduck-embodied/artifacts/week02/01_baseline_5999/raw/straight_030_seed42_episode000_steps.csv \
  --output ~/projects/microduck-embodied/results/week02/01_baseline_5999/figures/straight_030_model_5999.png \
  --smooth-window-s 1.0
```

图像属于可审阅的实验结果，可以提交 Git；逐步原始 CSV 继续保持本地忽略。

## 20-Episode 官方 Reset 小批量

该命令集使用官方 Velocity 基础环境的初始根状态范围：`x/y` 为
`[-0.5, 0.5] m`、`z` 为 `[0.12, 0.13] m`、`yaw` 为 `[-π, π]`。官方配置的
初始关节位置与速度扰动均为 0，因此这里不会额外编造关节扰动。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy logs/rsl_rl/velocity/2026-09-05_22-08-33_baseline-flat-resume-5000/microduck_velocity_flat_5999.onnx \
  --command-set ~/projects/microduck-embodied/evaluation/command_sets/straight_20_official_reset.json
```

汇总 JSON 会额外给出偏航方向计数、Fall Rate，以及偏航、速度、RMSE 和横向
位移的均值、中位数、总体标准差、最小值和最大值。

### 绘制 20-Episode 批次汇总图

批次图包含累计航向、对齐后的 XY 轨迹、最终偏航分布，以及 Episode 平均
`vx-wz` 关系。由于初始世界位置和朝向经过随机化，XY 轨迹会先转换到各自的
初始机体坐标系，再进行叠加和求平均。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/plot_locomotion_batch.py \
  --summary ~/projects/microduck-embodied/results/week02/01_baseline_5999/summaries/straight_030_official_reset_20_microduck_velocity_flat_5999.json \
  --output ~/projects/microduck-embodied/results/week02/01_baseline_5999/figures/straight_030_official_reset_20_batch.png \
  --smooth-window-s 1.0
```

## 8 项指令响应矩阵

在单一直行缺陷通过 20 个官方 reset 得到确认后，使用同一组种子 `42–61`
分别评测静止、慢速/正常前进、后退、正负横移和正负旋转。每项运行 20 个
Episode，共 160 个 Episode；相同 Episode 编号在各指令间共享初始状态，便于
成对比较。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy logs/rsl_rl/velocity/2026-09-05_22-08-33_baseline-flat-resume-5000/microduck_velocity_flat_5999.onnx \
  --command-set ~/projects/microduck-embodied/evaluation/command_sets/command_response_20.json
```

评测器会为每项指令写出独立 JSON，并额外生成一个跨指令 suite summary。该
矩阵用于判断单侧偏航是否存在于静止、前后运动、横移和正负旋转等不同命令
区域，不把它误当成只与正常前进有关的问题。

生成跨指令比较图：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/plot_command_response.py \
  --suite ~/projects/microduck-embodied/results/week02/01_baseline_5999/summaries/command_response_20_microduck_velocity_flat_5999.json \
  --output ~/projects/microduck-embodied/results/week02/01_baseline_5999/figures/command_response_20_model_5999.png
```

### 左右腿 Action 镜像诊断

使用官方 `symmetry.py` 的 61 维 Observation 和 14 维 Action 镜像规则，分别
计算 ONNX 本身的镜像等变误差，以及正负旋转、正负横移 Episode 的闭环成对
误差。默认每 5 个控制步抽取一次 Observation 重新推理，即以 10 Hz 采样检查
策略网络；闭环 Action 对比仍使用完整 50 Hz 数据。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/analyze_action_symmetry.py \
  --suite ~/projects/microduck-embodied/results/week02/01_baseline_5999/summaries/command_response_20_microduck_velocity_flat_5999.json \
  --output-json ~/projects/microduck-embodied/results/week02/03_action_symmetry/summaries/action_symmetry_model_5999.json \
  --output-figure ~/projects/microduck-embodied/results/week02/03_action_symmetry/figures/action_symmetry_model_5999.png
```

## 骨架校验

从官方仓库的 `uv` 环境运行：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy logs/rsl_rl/velocity/2026-09-05_22-08-33_baseline-flat-resume-5000/microduck_velocity_flat_5999.onnx \
  --validate-only
```

去掉 `--validate-only` 即可运行第一个无窗口 Episode：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy logs/rsl_rl/velocity/2026-09-05_22-08-33_baseline-flat-resume-5000/microduck_velocity_flat_5999.onnx
```

## 结果目录

```text
results/week02/
├── README.md
├── 01_baseline_5999/
├── 02_checkpoint_analysis/
├── 03_action_symmetry/
├── 04_symmetry_ab/
└── 05_random_400/

artifacts/week02/          # 原始 CSV 和中间产物，不提交 Git
```

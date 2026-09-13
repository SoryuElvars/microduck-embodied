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

结果：
`results/week02/01_baseline_5999/summaries/episode_return_8x5_model_5999.json`。
这组 40 Episode 是独立协议，不与 ONNX 固定指令的 160 Episode 或随机指令的
400 Episode 合并计数。

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

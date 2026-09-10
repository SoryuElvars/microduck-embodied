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

该部署形态评测不加载训练环境的 Reward Manager，因此汇总文件明确记录
`reward_available: false`。Reward 和 Episode Return 将在后续官方环境式评测中加入。

## 绘制单 Episode 诊断图

使用 `matplotlib` 将原始 CSV 转换为四联图：前进速度、偏航角速度、累计航向和
世界坐标系 XY 轨迹。浅色细线表示 50 Hz 原始数据，深色粗线表示默认 1 秒滑动
平均，虚线表示目标命令或目标轨迹。

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/plot_locomotion_episode.py \
  --input ~/projects/microduck-embodied/results/week02/raw/straight_030_seed42_episode000_steps.csv \
  --output ~/projects/microduck-embodied/results/week02/figures/straight_030_model_5999.png \
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
  --summary ~/projects/microduck-embodied/results/week02/summary/straight_030_official_reset_20_microduck_velocity_flat_5999.json \
  --output ~/projects/microduck-embodied/results/week02/figures/straight_030_official_reset_20_batch.png \
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
  --suite ~/projects/microduck-embodied/results/week02/summary/command_response_20_microduck_velocity_flat_5999.json \
  --output ~/projects/microduck-embodied/results/week02/figures/command_response_20_model_5999.png
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
├── raw/       # 逐控制步和逐 Episode 原始数据，不提交 Git
├── summary/   # 可提交的汇总指标
└── figures/   # 可提交的图表
```

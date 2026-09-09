# Locomotion Benchmark

本目录用于实现 MicroDuck Locomotion 的自动、可复现评测。评测代码属于
`microduck-embodied`，但继续复用官方 `microduck_rl` 的 MuJoCo 模型、BAM M6
执行器、Observation 构造和 ONNX 推理组件。

## 当前阶段：Phase 1 单 Episode

当前入口已经能够：

- 解析并校验版本化的速度命令集；
- 校验官方仓库、场景文件和 ONNX 是否存在；
- 固定随机种子和结果目录；
- 打印预期的 `61 obs -> 14 actions` 合约及 Episode 计划。
- 复用官方 MuJoCo、BAM M6、Observation 和 ONNX Policy；
- 无窗口执行 1 秒零命令预热和 10 秒固定命令；
- 将 61 维 Observation、14 维 Action、速度、位置和姿态写入逐步 CSV；
- 输出速度 RMSE、平均速度、净偏航、位移和跌倒代理指标。

该部署形态评测不加载训练环境的 Reward Manager，因此汇总文件明确记录
`reward_available: false`。Reward 和 Episode Return 将在后续官方环境式评测中加入。

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

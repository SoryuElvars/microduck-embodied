# Locomotion Benchmark

本目录用于实现 MicroDuck Locomotion 的自动、可复现评测。评测代码属于
`microduck-embodied`，但继续复用官方 `microduck_rl` 的 MuJoCo 模型、BAM M6
执行器、Observation 构造和 ONNX 推理组件。

## 当前阶段：Phase 0 骨架

当前入口已经能够：

- 解析并校验版本化的速度命令集；
- 校验官方仓库、场景文件和 ONNX 是否存在；
- 固定随机种子和结果目录；
- 打印预期的 `61 obs -> 14 actions` 合约及 Episode 计划。

它暂不运行 MuJoCo。下一步将在 `run_episode()` 中加入无窗口 BAM 推理循环，
首先复现 `vx=0.30, vy=0, wz=0` 下的 10 秒严重偏航。

## 骨架校验

从官方仓库的 `uv` 环境运行：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/locomotion_benchmark.py \
  --policy logs/rsl_rl/velocity/2026-09-05_22-08-33_baseline-flat-resume-5000/microduck_velocity_flat_5999.onnx \
  --validate-only
```

只有这一步通过后，才实现和验证真实 Episode 循环。

## 结果目录

```text
results/week02/
├── raw/       # 逐控制步和逐 Episode 原始数据，不提交 Git
├── summary/   # 可提交的汇总指标
└── figures/   # 可提交的图表
```

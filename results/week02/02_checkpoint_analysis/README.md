# Checkpoint 分析

本目录包含两个互补阶段：

1. [三指令快速扫描](./probe_3_commands.md)：比较 `model_5000`、`5250`、`5500`、
   `5750` 和 `5999` 在前进及正负转向上的表现。
2. [完整八指令对比](./comparison_8_commands.md)：对 `model_5500`、`5750` 和
   `5999` 进行更完整的方向响应比较。

两项实验的跨 checkpoint 汇总位于 `summaries/`，图表位于 `figures/`。逐
checkpoint、逐命令结果及原始 CSV 保存在本地
`artifacts/week02/02_checkpoint_analysis/runs/`。

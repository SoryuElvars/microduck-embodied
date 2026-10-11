# Week 05 / 02：速度感知的停稳到达

训练侧成功条件改为：机器人位于 `0.20 m` 成功圈内，最近 0.5 s 的平均平面速度不超过 `0.10 m/s`、平均角速度绝对值不超过 `0.20 rad/s`，并再连续保持 0.5 s。阈值来自 Classical 零命令站立轨迹的 P99，是仿真代理，不是实机静止标准。

旧模型的 live quick 表明：K2 在新条件下 `10/10` 达成停稳，K5 为 `0/10`，K10 为 `3/10`；失败主要表现为接近目标但无法持续满足低速条件，而不是跌倒。由此不能简单用旧模型结果选择 K，也不能放宽成功阈值。

在新成功条件下重训的 K5/K10 仍出现“接近成功圈前主动减速”的现象。随后引入单一的 near-goal speed shaping 密集项，而不改变动作边界、低层或 benchmark。该项让命令速度从远处 cruise speed 向成功圈内 goal speed 平滑下降；其作用是鼓励接近目标时减速，不是人为强制停车。具体选模结果见下一报告。

原始轨迹和重计分结果位于 `artifacts/week05/03_stable_arrival_rescore/` 与
`artifacts/week05/04_stable_live/`；新成功条件下的 K5/K10 重训评测分别位于
`artifacts/week05/05_stable_k5_retrain_eval/` 与 `artifacts/week05/06_stable_k10_retrain_eval/`。

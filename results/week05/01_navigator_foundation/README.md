# Week 05 / 01：RL Navigator 基础与初始 K 快筛

高层 PPO 的输入为 `[dx_body, dy_body, vx_body, vy_body, wz_body, previous_cmd_vx, previous_cmd_wz]`，输出为归一化的 `[vx, wz]`。经限幅和 rate limit 后得到 `[vx, 0, wz]`，再保持 K 个 20 ms 低层控制步。底层在本阶段冻结为 61 维 observation 到 14 维 joint action 的 `model_1500`。

速度边界保持 Week04 Classical contract：`vx∈[0,0.25] m/s`、`wz∈[-0.5,0.5] rad/s`，线/角加速度上限为 `0.5 m/s²`、`1.0 rad/s²`。Navigator 不读取 MuJoCo 内部状态，只消费 `RobotState + GoalState`。

`K=2/5/10` 均通过 `64 envs × 5 iterations` CUDA smoke；随后各自的 100-iteration quick 都在冻结的 10 条 Week04 场景达到 `10/10`、零跌倒/超时/invalid。K2/K5/K10 的完成时间中位数分别为 12.52/12.44/12.06 s，但到达后平面速度约 0.149/0.137/0.160 m/s，明显高于 Classical Constrained 的 0.077 m/s。

因此这一步只证明链路和基本到达可用，不用 10 集结果冻结 K；它直接暴露了“进入成功圈并不等于停稳”的缺口，并引出下一阶段的速度感知成功条件与 reward 设计。原始证据保留在 `artifacts/week05/02_rl_quick/`。

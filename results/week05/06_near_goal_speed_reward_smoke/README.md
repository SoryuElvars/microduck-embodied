# Week 05：近目标超速惩罚的 K10 冒烟测试

## 目的与单变量边界

本轮只新增一个 Reward term，检验“实际速度随接近目标而降低”的训练链路。成功条件、
progress/goal/smooth/fall/timeout Reward、K10、Goal curriculum、冻结低层模型和评测
协议均未修改；这不是模型能力训练或 benchmark 结论。

新项为实际平面速度的超速惩罚：

$$
r_{near-speed}=-0.10\,G(d)\left[
\max\left(0,\frac{v_{xy}-v_{ref}(d)}{0.10}\right)
\right]^2
$$

其中成功半径 $r_s=0.20\,m$，近目标区域边界为 $d_{near}=0.35\,m$：

$$
G(d)=\operatorname{clip}\left(\frac{d_{near}-d}{d_{near}-r_s},0,1\right),
\quad
v_{ref}(d)=0.08+(0.20-0.08)
\operatorname{clip}\left(\frac{d-r_s}{d_{near}-r_s},0,1\right).
$$

因此 $d\ge0.35\,m$ 时该项严格为零；速度不超过距离对应目标时也为零。它惩罚
“接近目标时仍超速”，而不为完全停车给正奖励，避免加强已经观察到的圈外停滞。

## 冒烟结果

在官方仓库、CUDA、seed 42、K10、`64 envs × 5 iterations` 下运行成功：采样、PPO
更新、每项 Reward 日志、checkpoint 和自动 ONNX 导出均正常。训练日志中该项从
`-0.0002` 至约 `-0.0247`，说明它已实际进入 Reward Manager，而非仅注册在配置里。
此数值会随未训练策略探索而变动，不能解释为 reward 设计效果。

- 训练目录：`/home/elvars/projects/microduck_rl/logs/rsl_rl/pointgoal_navigation_k10/2026-10-10_15-01-43_week05-near-speed-k10-smoke-seed42/`
- 自动 ONNX：`2026-10-10_15-01-43_week05-near-speed-k10-smoke-seed42.onnx`，SHA-256
  `f25ae76786cf02ae25178c6e4c83a0dc0ff2c112948be8f2b7859666ee9f94e9`。
- 最后 checkpoint：`model_4.pt`，SHA-256
  `57bd07d6ccd22de9dfa4f55adbad99742a8110438c16135a03edbda9334265d5`。
- 官方配置单测：`16 passed`；官方仓库 `git diff --check` 通过。

执行命令：

```bash
cd /home/elvars/projects/microduck_rl
uv run train Mjlab-PointGoal-Navigation-K10-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.run-name week05-near-speed-k10-smoke-seed42 \
  --agent.logger tensorboard
```

训练入口和输出路径正确不代表 K10 已学会进圈、停稳或到达后不离圈。下一步须由用户
启动独立的 `100 iterations` 快筛训练；完成后才按 Week04 冻结场景分别执行位置与
停稳 live quick，且不得将 smoke 模型纳入能力比较。

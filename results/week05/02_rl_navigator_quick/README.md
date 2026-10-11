# Week 05 RL Navigator：三组 K 的独立快筛

## 结论

`K=2 / 5 / 10` 的 `model_99` 均完成 10 个配对 quick Episode：各 10/10 达到
Week 04 的成功定义，0 跌倒、0 超时、0 无效状态。三组均通过冻结的 quick 安全 Gate。
K5 的路径效率与最终距离较好，但 K10 的到达时间略短；10 个 Episode 还不足以冻结 K。

更关键的是，到达后 2 秒内 RL 的平面速度中位数约为 `0.14–0.16 m/s`，没有表现出
Classical Controller 的停车效果。这一段在训练环境的到达终止之后，不受训练 Reward
直接约束。当前 benchmark 的 `success` 表示完成 0.5 秒半径内保持、随后 2 秒未跌倒，
并不代表到达后静止。Constrained 在同一 quick 上的到达后平面速度中位数约为
`0.077 m/s`，RL 三组更高。下一轮模型/奖励设计前应单独审视这个缺口；不得因 Return 或
10/10 到达而视为已具备实机停车能力。

| 策略 | 成功 | 跌倒/超时/无效 | 完成时间中位数 | 路径效率中位数 | 最终距离中位数 | 到达后速度中位数 |
|---|---:|---:|---:|---:|---:|---:|
| Classical Constrained（Week04 quick） | 10/10 | 0/0/0 | 13.53 s | 0.731 | 0.158 m | 0.077 m/s |
| RL K2 | 10/10 | 0/0/0 | 12.52 s | 0.727 | 0.078 m | 0.149 m/s |
| RL K5 | 10/10 | 0/0/0 | 12.44 s | 0.742 | 0.040 m | 0.137 m/s |
| RL K10 | 10/10 | 0/0/0 | 12.06 s | 0.718 | 0.128 m | 0.160 m/s |

K5 相对 K2，在 10 个配对场景中有 8 个完成更快、8 个路径效率更高；相对 K10，
有 4 个完成更快、7 个路径效率更高。K5 对 Classical Constrained 的配对完成时间
差中位数为 `-1.05 s`、路径效率差中位数为 `+0.011`。样本小，不能把差异解释为
统计稳定的优胜。

## 冻结输入与运行口径

- 高层 checkpoint：三组各自的 `model_99.pt` 自动导出 ONNX，来自 `64 envs ×
  100 iterations`、`seed=42`、仅 Stage 1 的训练；曲线见 W&B
  [K2](https://wandb.ai/soryu-harbin-institute-of-technology/mjlab_microduck/runs/8l1douig)、
  [K5](https://wandb.ai/soryu-harbin-institute-of-technology/mjlab_microduck/runs/2rdoz1z3)、
  [K10](https://wandb.ai/soryu-harbin-institute-of-technology/mjlab_microduck/runs/ugzc6h7l)。
- 冻结底层：`artifacts/week03/06_yaw_only_tracking_candidate/models/model_1500.onnx`，
  SHA-256 `533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d`。
- 场景：Week04 `week04_classical_v1.json` 的 `w04e000–w04e009`，manifest
  SHA-256 `b2d4c62121e0c1b4f290094079512c09cf7f6eaf8e1918622f12483c2292337c`。
  各场景含固定 reset seed、20 秒 timeout、0.20 m 成功半径、0.5 秒保持、2 秒
  到达后观测、1 秒预热。
- 高层 Observation：7 维；Action：2 维归一化输出；命令保持 K 个 20 ms 底层步。
  `K2/K5/K10` 分别为 25/10/5 Hz。速度/加速度边界与训练配置及 Week04 共享边界相同。
- quick 场景为 1.0–1.8 m、全方向，超出本轮 Stage 1 训练的主要 0.4–0.8 m、
  前方 ±45° 分布。因此本结果是阶段外快筛，不是正式泛化证据。
- 部署 ONNX 评测没有官方训练 Reward；这里只计算轨迹与任务指标。

三个模型 SHA-256 分别为 K2 `98bb4e2a3da3214151ef7fec0459eda189d63773b5eba1c4d01b22f7a2c413df`、
K5 `e9ed8cee52d6ab72ad3a4e21e06b523830b9144c8954c8984a992ef33e6ed9ee`、
K10 `3a972ccb8331149cb9430f517e098618e9f4b8eab674e13ef9cff24f65bce2b6`。
原始逐步 CSV、每场景进度和含官方 commit 的完整摘要保存在本地
`artifacts/week05/02_rl_quick/k{2,5,10}_frozen_eval/{smoke,quick}/`。首次适配器试跑
保存在不带 `_frozen_eval` 的目录；两版逐项汇总完全一致。最终采用 `_frozen_eval`
目录，因为它直接调用字节未改动的 Week04 `run_episode`，仅在每个 Episode 的局部
上下文中注入 RL Navigator 和对应 K 的更新频率。

## 复现

从 `/home/elvars/projects/microduck_rl` 运行：

```bash
uv run python /home/elvars/projects/microduck-embodied/evaluation/rl_pointgoal_quick.py \
  --high-policy logs/rsl_rl/pointgoal_navigation_k5/2026-09-28_22-39-23_week05-nominal-k5-seed42/2026-09-28_22-39-23_week05-nominal-k5-seed42.onnx \
  --k 5 \
  --output-dir /home/elvars/projects/microduck-embodied/artifacts/week05/02_rl_quick/k5_frozen_eval \
  --quick
```

入口支持 `--validate-only`、`--smoke`、`--quick`。同一模型与协议的 quick 可按
Episode 断点续跑；模型、评测代码、协议或官方 commit 变化时签名不匹配，会拒绝混用
旧进度。静态校验和一个前方场景 smoke 先于每组 10-Episode quick 完成。

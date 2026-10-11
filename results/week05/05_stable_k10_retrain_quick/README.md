# Week 05：停稳成功条件重训 K10 的导出与快筛

## 结论

K10 在新停稳成功条件下完成 `64 envs × 100 iterations` Stage 1 训练；`model_99.pt`
已自动导出为含 Observation Normalizer 的 ONNX（`obs[1,7] → actions[1,2]`，
图首两个算子为 Normalizer 的 `Sub/Div`）。使用与 K5 相同的 Week04 冻结
10 个 Episode：原位置到达口径 **6/10**，新的停稳到达 live 口径 **5/10**；
两者均 **0 跌倒、0 无效状态**，分别 4 和 5 次超时。只看冻结的安全 Gate，
跌倒率和无效状态率均通过；但任务成功率与到达后保持仍不足以进入正式评测。

| K10 策略 | 原位置 quick | 停稳 live quick | 停稳后 2 秒离圈 |
|---|---:|---:|---:|
| 旧成功条件的 `model_99` | 10/10 | 3/10 | 未在本报告重计 |
| 新成功条件的 `model_99` | **6/10** | **5/10** | **4/5** |

与新成功条件下 K5 的双口径 0/10 相比，K10 明显改善。但相对旧 K10，
停稳 live 从 3/10 升到 5/10 的同时，原位置到达从 10/10 降到 6/10；
10 个 Episode 只是配对快筛，不足以冻结 K 值或断言收益稳定。K10 的 5 次
停稳成功中，有 4 次在达成条件后继续观测的 2 秒内再次离开 `0.20 m` 成功圈。
当前 live 入口将“达成一次停稳门槛，且到达后观测期不跌倒、无无效状态”记为成功；
`redeparted_after_arrival` 另报，不把离圈自动改写为失败。因此 **5/10 不是
持续停稳 2 秒的成功率**。

超时的 5 个场景中，`w04e003/004` 的最近距离分别约 `0.201/0.205 m`，
没有进入圈；`w04e002/006/008` 最近达到约 `0.192/0.198/0.199 m`，但未能
连续满足完整停稳条件。末段 1 秒的平均 `cmd_vx` 约 `0.021–0.027 m/s`，实际
平面速度均值约 `0.064–0.070 m/s`。这支持“接近目标时命令明显减小”这一
可检验假设，但不能仅凭此认定策略有意在圈外停车，也不能排除低层跟踪、状态
估计或稀疏成功信号的作用。

## 冻结输入

- 官方 commit `062921c4c9f65107e391df042f8de424136213c3`，训练目录保存了
  当时未提交的 `git/microduck_rl.diff`。训练配置是 seed 42、K10、
  `num_steps_per_env=12`、100 iterations、W&B logger；与旧 K10 的保存配置
  相比，仅 run 名称及两个停稳阈值字段不同。训练曲线：
  [W&B K10 stable quick](https://wandb.ai/soryu-harbin-institute-of-technology/mjlab_microduck/runs/1gnybowv)。
- 训练目录：`/home/elvars/projects/microduck_rl/logs/rsl_rl/pointgoal_navigation_k10/2026-09-30_19-38-14_week05-stable-k10-quick-seed42/`。
  `model_99.pt` SHA-256：`2b2b4161b14cadc31e8d5b3300491099d67d577be7adf42d81d7014796ea50dd`；
  同目录 ONNX SHA-256：`aa4303196e7df2e0728e59524fcd89dd68d11665b8a2a8d5fd629f8fec7f64b4`。
- 冻结底层 ONNX SHA-256：`533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d`；
  Week04 Episode manifest SHA-256：`b2d4c62121e0c1b4f290094079512c09cf7f6eaf8e1918622f12483c2292337c`。
  场景为 `w04e000–w04e009`，1.0–1.8 m、全方向，超出 Stage 1 主要
  0.4–0.8 m 前方目标分布；两种口径均用 20 秒 timeout。
- 位置条件：`0.20 m` 圈内保持 `0.5 s`；停稳条件额外要求最近 `0.5 s`
  平面速率均值 `≤0.10 m/s`、有符号角速度均值的绝对值 `≤0.20 rad/s`，
  且该状态连续保持 `0.5 s`。到达后继续观测 2 秒。部署 ONNX 不提供训练 Reward。

## 执行记录与复现

官方配置单测 `15 passed`，个人仓库单测 `108 passed`，两个仓库
`git diff --check` 通过。两个评测入口各按 `--validate-only → --smoke →
--quick` 完成；smoke 不计入能力统计。位置和停稳 quick 的原始 CSV、签名与
逐场景摘要分别保存在本地 `artifacts/week05/06_stable_k10_retrain_eval/position_v1/`
及 `stable_v1/`，不提交 Git。
两入口的 10 个同场景导航轨迹在共同前缀上，时间、距离、命令、实际速度和世界
坐标逐步最大绝对差均为 `0`；差异从各自成功条件触发后的终止时机开始。

从 `/home/elvars/projects/microduck_rl` 运行：

```bash
uv run python /home/elvars/projects/microduck-embodied/evaluation/rl_pointgoal_quick.py \
  --high-policy logs/rsl_rl/pointgoal_navigation_k10/2026-09-30_19-38-14_week05-stable-k10-quick-seed42/2026-09-30_19-38-14_week05-stable-k10-quick-seed42.onnx \
  --k 10 --output-dir /home/elvars/projects/microduck-embodied/artifacts/week05/06_stable_k10_retrain_eval/position_v1 \
  --quick

uv run python /home/elvars/projects/microduck-embodied/evaluation/rl_pointgoal_stable_quick.py \
  --high-policy logs/rsl_rl/pointgoal_navigation_k10/2026-09-30_19-38-14_week05-stable-k10-quick-seed42/2026-09-30_19-38-14_week05-stable-k10-quick-seed42.onnx \
  --k 10 --output-dir /home/elvars/projects/microduck-embodied/artifacts/week05/06_stable_k10_retrain_eval/stable_v1 \
  --quick
```

本次只做快筛，不启动 formal/full，不更改 Reward、成功阈值、模型或评测协议。
下一步应先区分圈外停滞、短暂进圈但不能保持，以及已达成停稳后再次离圈的
轨迹；再决定单因素实验。不能直接把这 5 次达成解释成可靠停车能力。

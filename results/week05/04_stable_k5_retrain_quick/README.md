# Week 05：停稳成功条件重训 K5 的导出与快筛

## 结论

新成功条件下的 K5 已完成 `64 envs × 100 iterations` Stage 1 快筛训练；训练结束时
自动导出了 `model_99.pt` 对应的 ONNX，包含 Observation Normalizer。该 ONNX
在 Week04 冻结的 10 个配对场景中，原位置到达口径和新的停稳到达口径均为
**0/10 成功、0 跌倒、10 超时**。两种评测的最终距离中位数均为 `0.221 m`。
所有场景的最小距离为 `0.208–0.224 m`，均未进入 `0.20 m` 成功圈；因此这次
快筛首先暴露的是**圈外减速/停滞**，不是进圈后才被速度门槛挡住。

与同协议旧 K5 对照：旧模型在原位置口径为 10/10、最终距离中位数 `0.040 m`；
在新停稳口径下 live quick 为 0/10，但最终距离中位数约 `0.147 m`。新旧两轮
训练均为 seed 42、100 iterations、Stage 1；保存配置中除 run 名称和两个停稳
阈值字段外无差异。这个配对快筛证明新模型在当前阶段外场景明显退化，但不能单凭
10 条轨迹断言具体是哪一项 Reward、成功条件或训练时长导致。训练分布主要是
`0.4–0.8 m`、前方目标；冻结 quick 是 `1.0–1.8 m`、全方向，不能把训练日志中的
非零 goal 项当作该 quick 的成功率。

| K5 策略 | Week04 原位置 quick | 新停稳 live quick | 跌倒/超时（新模型） |
|---|---:|---:|---:|
| 旧成功条件训练的 `model_99` | 10/10 | 0/10 | — |
| 新成功条件训练的 `model_99` | **0/10** | **0/10** | **0/10** |

快筛未通过，不升级到 formal/full，也不据此放宽阈值或修改奖励。下一步应先对照
训练 Stage 1 场景与圈外末段轨迹，形成**单因素**改动假设，再由用户决定是否继续训练。

## 冻结输入与导出核验

- 官方 commit：`062921c4c9f65107e391df042f8de424136213c3`；运行时保留了
  官方工作区的未提交修改，训练目录中的 `git/microduck_rl.diff` 保存了当时差异。
- 训练目录：`/home/elvars/projects/microduck_rl/logs/rsl_rl/pointgoal_navigation_k5/2026-09-30_18-34-30_week05-stable-k5-quick-seed42/`。
  `params/agent.yaml` 记录 seed 42、64 envs、24 steps/env、100 iterations、
  W&B logger；训练曲线为 [W&B K5 stable quick](https://wandb.ai/soryu-harbin-institute-of-technology/mjlab_microduck/runs/viezaagl)。
- `model_99.pt` SHA-256：`c0fbb582544bffb03ef35854f4096a9615c7e7406f83cfdbac33c4bc371e3d35`。
  同目录自动导出的 `2026-09-30_18-34-30_week05-stable-k5-quick-seed42.onnx`
  SHA-256：`9e17a7281609e05b621e8ac7e69ca215a6eebfa7f47d89927f953bbe7ec38de4`。
  ONNX 输入 `obs[1,7]`、输出 `actions[1,2]`，首层含 `Sub/Div` Normalizer；
  有限值探针通过。部署端仍按冻结命令边界裁剪动作。
- 底层 ONNX SHA-256：`533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d`；
  Week04 Episode manifest SHA-256：`b2d4c62121e0c1b4f290094079512c09cf7f6eaf8e1918622f12483c2292337c`。
- 位置口径：半径 `0.20 m` 内连续保持 `0.5 s`；停稳口径再要求最近 `0.5 s`
  平面速率均值不超过 `0.10 m/s`、有符号角速度均值绝对值不超过
  `0.20 rad/s`，并连续保持 `0.5 s`。两者均为 20 秒导航 timeout。

## 评测与复现

官方配置单测 `15 passed`；个人仓库单测 `108 passed`；两个仓库的
`git diff --check` 均通过。两个评测入口均完成 `--validate-only → --smoke →
--quick`；smoke 只验证加载、运行和记录，不计入能力分数。逐步 CSV 与签名摘要
分别在本地 `artifacts/week05/05_stable_k5_retrain_eval/position_v1/` 和
`stable_v1/`，原始数据不提交 Git。部署 ONNX 没有训练 Reward，本报告不把轨迹
指标称作 Episode Return。

从 `/home/elvars/projects/microduck_rl` 运行；下列 `--quick` 可按签名断点续跑：

```bash
uv run python /home/elvars/projects/microduck-embodied/evaluation/rl_pointgoal_quick.py \
  --high-policy logs/rsl_rl/pointgoal_navigation_k5/2026-09-30_18-34-30_week05-stable-k5-quick-seed42/2026-09-30_18-34-30_week05-stable-k5-quick-seed42.onnx \
  --k 5 --output-dir /home/elvars/projects/microduck-embodied/artifacts/week05/05_stable_k5_retrain_eval/position_v1 \
  --quick

uv run python /home/elvars/projects/microduck-embodied/evaluation/rl_pointgoal_stable_quick.py \
  --high-policy logs/rsl_rl/pointgoal_navigation_k5/2026-09-30_18-34-30_week05-stable-k5-quick-seed42/2026-09-30_18-34-30_week05-stable-k5-quick-seed42.onnx \
  --k 5 --output-dir /home/elvars/projects/microduck-embodied/artifacts/week05/05_stable_k5_retrain_eval/stable_v1 \
  --quick
```

位置口径的逐场景最小距离均大于半径，因此速度门槛在本轮没有成为决定性条件。
这与训练端记录到的部分 Stage 1 Goal 达成不矛盾：两者目标分布不同。对下一轮
最有价值的诊断是检查接近 `0.20 m` 时的 `cmd_vx`、实际速度与距离曲线，并用
训练分布内的独立小样本检验是否同样出现圈外停滞；在明确假设前不追加训练轮次。

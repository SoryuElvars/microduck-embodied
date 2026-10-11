# Week 05：近目标超速惩罚 K10 快筛评测

## 结论

在新停稳条件 K10 的基础上，唯一新增“近目标实际平面速度超速惩罚”后，`64 envs ×
100 iterations` 的模型在 Week04 冻结的 10 个配对场景上达到：**位置到达 10/10，
停稳 live 到达 10/10，0 跌倒、0 超时、0 无效状态**。10 个停稳成功都在额外 2 秒
观测期内保持在 `0.20 m` 成功圈，`redeparted_after_arrival=0/10`。

| K10，100 iterations，seed 42 | 位置 quick | 停稳 live quick | 跌倒/超时 | 到达后离圈 |
|---|---:|---:|---:|---:|
| 新停稳条件，未加速度项 | 6/10 | 5/10 | 0/4；0/5 | 位置：2/6；停稳：4/5 |
| 新停稳条件 + 近目标超速惩罚 | **10/10** | **10/10** | **0/0；0/0** | **0/10；0/10** |

位置 quick 的完成时间中位数为 `14.18 s`，路径效率中位数为 `0.689`，最终距离中位数
为 `0.183 m`。相较未加速度项的 6 个成功样本，路径效率的成功样本中位数从 `0.717`
下降到 `0.689`；但两组的成功集合不同，不能把这个小差异解释为确定的效率退化。
新模型停稳完成时间中位数为 `14.2 s`。到达后 2 秒的每场景平均实际平面速度中位数为
`0.070 m/s`，低于 `0.10 m/s` 门槛；瞬时速度中位数最大值约 `0.124 m/s` 并不违反
训练/评测的**窗口平均**速度定义。10 条轨迹的到达后最大目标距离最高为 `0.199 m`。

这是固定 10 场景 quick 的单因素结果，不是 formal、OOD 或 Sim2Real 结论。它支持
“近目标超速惩罚改善本协议下的入圈、停稳和短时保持”这一假设；仍需在用户启动的
formal 前冻结本模型、训练配置、场景、seeds 和 Gate，且正式结果不得与 quick 合并。

## 单变量定义与冻结输入

新增项的未加权输出为：

$$
p(d,v_{xy})=G(d)\left[\max\left(0,\frac{v_{xy}-v_{ref}(d)}{0.10}\right)\right]^2,
\qquad r_{near-speed}=-0.10p.
$$

成功半径 $r_s=0.20\,m$，近目标边界 $d_{near}=0.35\,m$；$G(d)$ 从边界外的零
线性增至成功圈处的一，参考速度从 `0.20 m/s` 线性降至 `0.08 m/s`。它只惩罚
实际速度超过参考值的部分，不对圈外停车给正奖励。训练配置与上一轮 K10 的差异仅为
该 Reward term；成功条件仍是半径 `0.20 m`、最近 `0.5 s` 平面速率均值
`≤0.10 m/s`、有符号角速度均值绝对值 `≤0.20 rad/s`、再连续保持 `0.5 s`。

- 官方 commit：`062921c4c9f65107e391df042f8de424136213c3`；训练目录保存了当时的
  `git/microduck_rl.diff`。
- 训练：K10、seed 42、64 envs、12 steps/env、100 iterations；[W&B 曲线](https://wandb.ai/soryu-harbin-institute-of-technology/mjlab_microduck/runs/3f5yvucw)。
- 高层 ONNX：`/home/elvars/projects/microduck_rl/logs/rsl_rl/pointgoal_navigation_k10/2026-10-10_15-04-22_week05-near-speed-k10-quick-seed42/2026-10-10_15-04-22_week05-near-speed-k10-quick-seed42.onnx`，
  SHA-256 `275c0ddae09154c1c497ccbce00d9571d6f06e162fdab392ac5c6000e5c9dcc8`。
- 高层 checkpoint `model_99.pt` SHA-256：`1e649e56dd10ad122d172bdcde149ebcd2e88ed39541e580583fe2cd040133fa`。
- 冻结低层 ONNX SHA-256：`533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d`；
  Week04 manifest SHA-256：`b2d4c62121e0c1b4f290094079512c09cf7f6eaf8e1918622f12483c2292337c`。

Quick 场景为 `w04e000–w04e009`，距离 1.0–1.8 m、全方向；训练 Stage 1 主要为
0.4–0.8 m、前方目标。因此 quick 代表一个冻结阶段外筛查，而不是训练分布内成绩。

## 验证与复现

官方配置单测 `16 passed`，个人仓库单测 `108 passed`，两个仓库 `git diff --check`
通过。位置与停稳入口均按 `--validate-only → --smoke → --quick` 执行。两入口对
10 个场景的共同导航前缀中，时间、距离、命令、实际速度和世界坐标逐步最大绝对差
均为零；不同仅从位置成功先终止、而停稳入口继续控制的位置开始。

原始 CSV、逐场景进度与含签名的摘要保存在本地
`artifacts/week05/07_near-speed_k10_quick_eval/{position_v1,stable_v1}/`，不提交 Git。
部署 ONNX 没有训练 Reward，本报告不将其轨迹指标称为 Episode Return。

从 `/home/elvars/projects/microduck_rl` 复现：

```bash
uv run python /home/elvars/projects/microduck-embodied/evaluation/rl_pointgoal_quick.py \
  --high-policy logs/rsl_rl/pointgoal_navigation_k10/2026-10-10_15-04-22_week05-near-speed-k10-quick-seed42/2026-10-10_15-04-22_week05-near-speed-k10-quick-seed42.onnx \
  --k 10 \
  --output-dir /home/elvars/projects/microduck-embodied/artifacts/week05/07_near-speed_k10_quick_eval/position_v1 \
  --quick

uv run python /home/elvars/projects/microduck-embodied/evaluation/rl_pointgoal_stable_quick.py \
  --high-policy logs/rsl_rl/pointgoal_navigation_k10/2026-10-10_15-04-22_week05-near-speed-k10-quick-seed42/2026-10-10_15-04-22_week05-near-speed-k10-quick-seed42.onnx \
  --k 10 \
  --output-dir /home/elvars/projects/microduck-embodied/artifacts/week05/07_near-speed_k10_quick_eval/stable_v1 \
  --quick
```

不在本轮自动启动 formal/full。若进入正式评测，应先由用户明确启动，并保持以上 ONNX、
协议、限定、seeds、二秒保持诊断和 Gate 完全冻结。

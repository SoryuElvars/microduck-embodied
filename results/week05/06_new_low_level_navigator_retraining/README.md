# Week 05 / 06：新低层上的 Navigator 重训与 checkpoint quick

本报告只讨论冻结低层 `model_1498.onnx` 上重新训练的高层 Navigator。它不继承旧低层
`model_1500.onnx` 上 `model_1200` 的 formal 结论，也不把 smoke、quick 与 formal 的 Episode
合并。

冻结低层 ONNX SHA-256 为
`44f61b6653a75506f52b68e890b75156e34ae68a8db9296102c575adeb558e65`。部署评测继续使用冻结
`week04_classical_v1`：目标距离 `1.0–1.8 m`、相对方位覆盖 `[-π, π]`，quick 固定为 10 个
EpisodeSpec，导航 timeout 为 20 s，到达保持 0.5 s，成功后继续观察 2 s。协议 manifest SHA-256
为 `b2d4c62121e0c1b4f290094079512c09cf7f6eaf8e1918622f12483c2292337c`，评测记录的官方仓库
commit 为 `5b052cc74728ca55cf19261e2b194806d1e991d2`。训练 Stage 1 则只到 `0.8 m`、90% 位于机体前方 `±45°`；因此仅完成 Stage 1 的
checkpoint 在 Week04 quick 上属于跨分布诊断，不是正式验收。

## 被废弃的 1-env 横向比较

最初的新低层 K2/K5/K10 100-iteration 训练分别使用 `num_envs=1`，每 iteration 的 PPO 样本数为：

| K | `num_steps_per_env` | 每 iteration 样本数 |
|---|---:|---:|
| K2 | 60 | 60 |
| K5 | 24 | 24 |
| K10 | 12 | 12 |

因此这三次运行同时改变了 K 和 PPO batch size，不能据此得出“K 越小越难学”的横向结论。
`artifacts/week05/21_low1498_k10_quick_training/`、`23_low1498_k5_quick_training/` 与
`24_low1498_k_screen_training/` 只保留为历史诊断；`22_low1498_k10_quick_eval/` 的全难度
`0/10` 也只对应未完成课程的早期模型。

`artifacts/week05/25_low_level_bridge_parity/` 中固定命令旁路高层策略的证据仍有效：同一冻结低层
在 K2/K5/K10 下都能稳定执行 `vx=0.25 m/s`，没有发现 observation、归一化、符号或 rate-limit
桥接错误。失效的是 1-env PPO 横向能力比较，不是该低层执行链诊断。

## K10：64-env、完整三阶段训练

训练目录：

```text
/home/elvars/projects/microduck_rl/logs/rsl_rl/pointgoal_navigation_k10/
2026-10-11_15-23-07_week05-low1498-k10-curr500-formal-seed42
```

配置为 64 environments、seed 42、1500 iterations、每阶段 500 iterations。Stage 1/2/3
分别从 iteration 0/500/1000 开始；三个阶段约覆盖相同的 500 × 9.6 s 物理仿真时间。目录名中的
`formal` 是训练 run label，不代表已执行 formal 评测。

16 个 checkpoint 均完成独立的 1-Episode smoke 和 10-Episode quick：

| checkpoint | quick 成功 | timeout | fall | 中位终点距离 | 到达样本中位完成时间 | 中位停稳漂移 | 失败 Episode |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0/10 | 10 | 0 | 1.370 m | — | — | 全部 |
| 100 | 8/10 | 2 | 0 | 0.168 m | 15.64 s | 0.025 m | w04e002, w04e009 |
| 200 | 8/10 | 2 | 0 | 0.178 m | 16.33 s | 0.017 m | w04e002, w04e009 |
| 300 | 8/10 | 2 | 0 | 0.192 m | 17.22 s | 0.008 m | w04e002, w04e009 |
| 400 | 8/10 | 2 | 0 | 0.180 m | 14.92 s | 0.014 m | w04e002, w04e009 |
| 500 | 8/10 | 2 | 0 | 0.169 m | 14.92 s | 0.022 m | w04e002, w04e009 |
| 600 | 8/10 | 2 | 0 | 0.171 m | 14.80 s | 0.022 m | w04e002, w04e009 |
| 700 | 8/10 | 2 | 0 | 0.175 m | 14.68 s | 0.019 m | w04e002, w04e009 |
| 800 | 8/10 | 2 | 0 | 0.177 m | 15.24 s | 0.018 m | w04e002, w04e009 |
| 900 | 8/10 | 2 | 0 | 0.179 m | 15.73 s | 0.016 m | w04e002, w04e009 |
| 1000 | 8/10 | 2 | 0 | 0.171 m | 15.22 s | 0.023 m | w04e002, w04e009 |
| 1100 | 8/10 | 2 | 0 | 0.178 m | 15.65 s | 0.017 m | w04e002, w04e009 |
| 1200 | 8/10 | 2 | 0 | 0.177 m | 15.51 s | 0.017 m | w04e002, w04e009 |
| 1300 | 8/10 | 2 | 0 | 0.166 m | 14.87 s | 0.027 m | w04e002, w04e009 |
| 1400 | 8/10 | 2 | 0 | 0.165 m | 14.70 s | 0.026 m | w04e002, w04e009 |
| 1499 | 8/10 | 2 | 0 | 0.169 m | 15.01 s | 0.023 m | w04e002, w04e009 |

所有 checkpoint 都是 0 fall、0 invalid；quick Gate 只检查 fall/invalid，因此“通过 Gate”不等于
checkpoint 已被选中。`model_100–1499` 的成功数完全相同，当前 10-Episode quick 无法可靠区分最佳
checkpoint。`model_1400` 的中位终点距离最小，但停稳漂移反而较大，不能据此单独冻结。

本批 K10 checkpoint-sweep 原始证据位于
`artifacts/week05/26_low1498_curr500_checkpoint_sweep/`。共计 16 个 smoke 和 16 个 quick
summary；该目录没有 `run_mode=full` 结果，后续冻结 `model_1499` 的 formal 见本文末节。

## K2/K5：统一 64-env 的 Stage 1 对照

为消除 batch size 混杂，K2/K5 使用与 K10 相同的 64 environments、seed 42 和 500 Stage-1
iterations 重新训练。评测只导出了 iteration 200/300/400/499：

```text
K2: /home/elvars/projects/microduck_rl/logs/rsl_rl/pointgoal_navigation_k2/
    2026-10-11_18-27-03_week05-low1498-k2-curr500-compare-seed42
K5: /home/elvars/projects/microduck_rl/logs/rsl_rl/pointgoal_navigation_k5/
    2026-10-11_18-51-59_week05-low1498-k5-curr500-compare-seed42
```

| K | checkpoint | 训练 entropy | quick 成功 | timeout | fall | 中位终点距离 | 中位完成时间 | 中位停稳漂移 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K2 | 200 | 2.311 | 8/10 | 2 | 0 | 0.170 m | 15.65 s | 0.023 m |
| K2 | 300 | 3.018 | 7/10 | 3 | 0 | 0.177 m | 17.22 s | 0.020 m |
| K2 | 400 | 3.365 | 5/10 | 5 | 0 | 0.196 m | 19.12 s | 0.020 m |
| K2 | 499 | 3.883 | 0/10 | 10 | 0 | 0.224 m | — | — |
| K5 | 200 | 1.522 | 7/10 | 3 | 0 | 0.171 m | 14.66 s | 0.024 m |
| K5 | 300 | 1.479 | 7/10 | 3 | 0 | 0.180 m | 16.28 s | 0.014 m |
| K5 | 400 | 1.321 | 8/10 | 2 | 0 | 0.183 m | 15.26 s | 0.012 m |
| K5 | 499 | 1.090 | 8/10 | 2 | 0 | 0.176 m | 15.39 s | 0.019 m |

K2 的 `Policy/mean_std` 从 iteration 200 的 0.772 增至 499 的 1.749，训练 entropy 与部署 quick
退化同步；到 499 时十个目标均在成功圈外 timeout。ONNX 部署使用确定性策略均值，因此 entropy
不会直接向部署动作加噪声；这里只能确认训练不稳定与性能退化相关，不能把 entropy 单独认定为根因。
本轮决定停止 K2，不继续投入 Stage 2/3。

K2/K5 Stage-1 原始结果位于
`artifacts/week05/27_low1498_k2_k5_curr500_checkpoint_screen/`。这批结果没有 formal。

## K5：续训完成后的 16-checkpoint quick

K5 从 Stage-1 `model_499.pt` 恢复，课程计数没有重置。续训目录为：

```text
/home/elvars/projects/microduck_rl/logs/rsl_rl/pointgoal_navigation_k5/
2026-10-11_19-28-13_week05-low1498-k5-curr500-resume500-to1500-seed42
```

配置继续使用 64 environments、seed 42；`model_500` 的 `common_step_counter=12024`，最终
`model_1499` 为 `36000`，因此 Stage 2/3 已按原课程时间轴完成。对齐 K10 的 16 个 checkpoint
由首段训练的 `0–400` 和续训的 `500–1499` 组成。全部完成独立静态校验、1-Episode smoke 和
10-Episode quick：

| checkpoint | quick 成功 | timeout | fall | 中位终点距离 | 到达样本中位完成时间 | 中位停稳漂移 | 失败 Episode |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0/10 | 10 | 0 | 1.839 m | — | — | 全部 |
| 100 | 5/10 | 5 | 0 | 0.189 m | 18.24 s | 0.017 m | w04e000, w04e002, w04e006, w04e008, w04e009 |
| 200 | 7/10 | 3 | 0 | 0.171 m | 14.66 s | 0.024 m | w04e002, w04e006, w04e009 |
| 300 | 7/10 | 3 | 0 | 0.180 m | 16.28 s | 0.014 m | w04e000, w04e002, w04e009 |
| 400 | 8/10 | 2 | 0 | 0.183 m | 15.26 s | 0.012 m | w04e002, w04e009 |
| 500 | 8/10 | 2 | 0 | 0.179 m | 15.53 s | 0.018 m | w04e002, w04e009 |
| 600 | 7/10 | 3 | 0 | 0.184 m | 15.56 s | 0.013 m | w04e000, w04e002, w04e009 |
| 700 | 8/10 | 2 | 0 | 0.176 m | 15.42 s | 0.021 m | w04e002, w04e009 |
| 800 | 8/10 | 2 | 0 | 0.180 m | 15.44 s | 0.016 m | w04e002, w04e009 |
| 900 | 8/10 | 2 | 0 | 0.180 m | 14.07 s | 0.016 m | w04e002, w04e009 |
| 1000 | 8/10 | 2 | 0 | 0.173 m | 14.63 s | 0.021 m | w04e002, w04e009 |
| 1100 | 8/10 | 2 | 0 | 0.173 m | 14.94 s | 0.020 m | w04e002, w04e009 |
| 1200 | 8/10 | 2 | 0 | 0.176 m | 14.99 s | 0.018 m | w04e002, w04e009 |
| 1300 | 8/10 | 2 | 0 | 0.186 m | 15.59 s | 0.011 m | w04e002, w04e009 |
| 1400 | 8/10 | 2 | 0 | 0.178 m | 15.04 s | 0.016 m | w04e002, w04e009 |
| 1499 | 8/10 | 2 | 0 | 0.179 m | 14.70 s | 0.015 m | w04e002, w04e009 |

16 个 K5 checkpoint 均为 0 fall、0 invalid、0 post-arrival redeparture。学习速度慢于 K10：K10
从 iteration 100 起就是 8/10，而 K5 到 iteration 400 才首次达到 8/10，并在 600 短暂回落到
7/10。进入 Stage 2 后期后，K5 的 `700–1499` 与 K10 在这组固定 EpisodeSpec 上均为 8/10，且
失败集合完全相同，都是最远距离的 `w04e002` 与 `w04e009`。

成熟 checkpoint 的次级指标没有给出单一赢家：K5 `model_900` 的到达样本中位完成时间最短
（14.07 s），`model_1000` 的中位终点距离最小（0.173 m），`model_1300` 的停稳漂移最小
（0.011 m）；K10 后期 `model_1400` 的中位终点距离更小（0.165 m），但停稳漂移更大
（0.026 m）。这些差异来自每模型仅 10 个固定 Episode，不能据此宣称 K5 或 K10 总体更优。

K5 全课程原始结果位于
`artifacts/week05/28_low1498_k5_curr1500_checkpoint_sweep/`。在该 checkpoint sweep 阶段尚无
`formal/full` 结果，因此先冻结了候选与比较规则，没有把 quick 的 8/10 当成正式成功率；随后执行的
配对 formal 单独记录如下。

## K5 model_1499 与 K10 model_1499：200-Episode formal

在 checkpoint、协议和输出目录冻结后，对同为 iteration 1499、training seed 42 的 K5/K10
运行了共享 200 EpisodeSpec 的 `--full`。两组结果使用相同低层、协议、manifest 和官方 commit：

- 低层：`model_1498.onnx`，SHA-256
  `44f61b6653a75506f52b68e890b75156e34ae68a8db9296102c575adeb558e65`；
- 协议：`week04_classical_v1`，manifest SHA-256
  `b2d4c62121e0c1b4f290094079512c09cf7f6eaf8e1918622f12483c2292337c`；
- 官方 commit：`5b052cc74728ca55cf19261e2b194806d1e991d2`；
- 每组均有 200 个原始轨迹、200 个进度记录和完整 `run_mode=full` summary。

| 指标 | K5 model_1499 | K10 model_1499 |
|---|---:|---:|
| Success | **175/200（87.5%）** | 169/200（84.5%） |
| Success 95% Wilson CI | 82.20%–91.39% | 78.84%–88.86% |
| Timeout | **25/200（12.5%）** | 31/200（15.5%） |
| Fall / Invalid | 0 / 0 | 0 / 0 |
| Post-arrival redeparture | 0/175 | 0/169 |
| 中位终点距离 | 0.179 m | **0.170 m** |
| 到达样本中位完成时间 | **16.26 s** | 16.38 s |
| 到达样本中位路径效率 | **0.753** | 0.741 |
| 中位停稳漂移 | **0.016 m** | 0.023 m |
| 中位最大漂移 | **0.016 m** | 0.023 m |
| 平均 `vx` 饱和比例 | 0.731 | **0.726** |
| 平均 `wz` 饱和比例 | 0.160 | **0.159** |

配对列联结果为：两者都成功 169、仅 K5 成功 6、仅 K10 成功 0、两者都失败 25。K10 的 169
个成功 Episode 是 K5 成功集合的严格子集；事后精确 McNemar 双侧检验 `p=0.03125`。因此在这组冻结
EpisodeSpec 上，K5 的 3 个百分点成功率优势不是由不同场景抽样造成的。不过置信区间与检验只描述
这两个固定 checkpoint 的部署表现，不包含训练 seed 方差，不能据此声称 K5 架构普遍优于 K10。

按目标距离分桶：

| 初始目标距离 | Episode | K5 成功 | K10 成功 | K5-only |
|---|---:|---:|---:|---:|
| 1.0–1.2 m | 52 | 52 | 52 | 0 |
| 1.2–1.4 m | 52 | 52 | 51 | 1 |
| 1.4–1.6 m | 49 | 44 | 44 | 0 |
| 1.6–1.8 m | 47 | **27** | 22 | 5 |

按相对目标方位四象限分桶：

| 方位 | Episode | K5 成功 | K10 成功 | K5-only |
|---|---:|---:|---:|---:|
| front | 44 | **44** | 43 | 1 |
| left | 61 | 51 | 51 | 0 |
| right | 49 | **46** | 42 | 4 |
| behind | 46 | **34** | 33 | 1 |

距离与方位是两种重叠切片，不能把差额相加。K5-only 的 6 个 Episode 为 `w04e049`、
`w04e066`、`w04e081`、`w04e105`、`w04e118`、`w04e157`；差异主要集中在远距离目标，并在右侧
目标上较多。两者共同成功的 169 个 Episode 中，K5 相对 K10 的配对中位差为：完成时间
`-0.26 s`、路径效率 `+0.0126`、停稳漂移 `-0.0069 m`。K10 的终点距离略小，但 K5 在成功率、
路径效率和到达后停稳上占优。

formal 原始证据位于 `artifacts/week05/29_low1498_k5_k10_formal/`。本轮可以把 K5
`model_1499` 作为这两个固定候选中的 nominal 首选；下一步若要回答 K 值本身是否更优，需要增加
独立训练 seeds，而不是继续从同一 seed 的 checkpoint 中挑选。

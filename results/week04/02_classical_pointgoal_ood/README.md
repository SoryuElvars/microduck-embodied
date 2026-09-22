# Week 4 Classical PointGoal OOD：胜出 Constrained Controller

## 当前结论

本目录记录 Week 4 胜出 Classical Controller 的轻量、单因素 OOD 证据。所有
条件固定同一个 `model_1500.onnx`、同一个 Constrained Go-to-Goal Controller、同一份
冻结 PointGoal manifest 和配对 reset seed；每个条件只改变一个部署侧因素。ONNX 部署评测
不加载训练 Reward，因此这里不报告或伪造 Episode Return。

三项计划内条件均已完成各自独立的 25-episode full。修复后的 Low Friction 中，地面、左脚
和右脚的滑动摩擦均在 MuJoCo 运行时读回为 `0.3`。三项条件都保持 25/25 任务成功、零
fall/timeout/invalid/post-arrival re-departure，但都改变了到达后的停止漂移。不同扰动条件
仍不合并为一个“整体成功率”，因为它们不是同一个测试分布。

## 冻结输入

| 项目 | 当前值 |
|---|---|
| OOD protocol | `evaluation/pointgoal_protocols/week04_classical_ood_v1.json` |
| OOD protocol 状态 / SHA-256 | `frozen` / `e695f028bcdacaa7b326ad87eff726f91e727cadc8034235dc03e5df60dee84a` |
| 基础 PointGoal protocol SHA-256 | `69db75465d5958f6c41ff49691cec0ff849cc6c5521ae4c28264f2806aec5be7` |
| Episode manifest SHA-256 | `b2d4c62121e0c1b4f290094079512c09cf7f6eaf8e1918622f12483c2292337c` |
| Controller | `constrained`，`[vx, 0, wz]`，`goal_tolerance=0.15 m` |
| ONNX / SHA-256 | `model_1500.onnx` / `533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d` |
| `microduck_rl` commit | `062921c4c9f65107e391df042f8de424136213c3` |
| Nominal 比较来源 | Week 4 Classical full 的同 EpisodeSpec 配对结果 |

## 条件覆盖与证据等级

| 条件 | 实际运行时注入 | 当前层级 | Episode | Gate | 结果边界 |
|---|---|---|---:|---|---|
| Low Friction 0.3 | `floor`、`left_foot_collision`、`right_foot_collision` 均读回 `0.3` | full | 25 | 通过 | 正式、配对的单条件 OOD 结果 |
| 40 ms Delay | `actuator_delay_steps=2` | full | 25 | 通过 | 正式、配对的单条件 OOD 结果 |
| 80% Motor | `bam_voltage_scale=0.8`、`bam_vin=5.92 V` | full | 25 | 通过 | 正式、配对的单条件 OOD 结果 |

各条件都独立报告，不能把 `25 + 25 + 25` 汇为 75 个同质 OOD Episode，也不能合并成功率。

## 修复后的 Low Friction full

Full 使用 `w04e010--w04e034`，与 quick 的 `w04e000--w04e009` 不重叠。正式源汇总的
run signature 为 `3b613b90fd91cdfd9d0027238b80205d9577c6b7c90e02467aea3d601cad63f9`。

| 指标 | Nominal，同 25 场景 | Low Friction 0.3 | 配对观察 |
|---|---:|---:|---|
| Success | 25/25 | 25/25 | 25 场均平局 |
| Fall / Timeout / Invalid | 0 / 0 / 0 | 0 / 0 / 0 | 安全 gate 通过 |
| Post-arrival Re-departure | 0/25 | 0/25 | gate 通过 |
| Median Final Distance | 0.1572 m | 0.1526 m | 配对中位差 `-0.0051 m` |
| Median Path Length to Arrival | 1.8382 m | 1.5726 m | 配对中位差 `-0.2745 m` |
| Median Path Efficiency | 0.7422 | 0.8685 | 配对中位差 `+0.1266` |
| Median Completion Time | 12.58 s | 12.98 s | 配对中位差 `+0.12 s` |
| Median Post-arrival Stop Drift | 0.0121 m | 0.0195 m | 配对中位差 `+0.0084 m`；Nominal 在 23/25 场更小 |
| Median Post-arrival Max Drift | 0.0276 m | 0.0278 m | 配对中位差 `-0.0003 m` |
| Median Post-arrival Max Goal Distance | — | — | Low Friction 在 25/25 场更大，配对中位差 `+0.0030 m` |

Low Friction 改变了运动轨迹：路径更短、路径效率更高，但完成时间略增加，且到达后的
净停止漂移从约 `1.21 cm` 增至 `1.95 cm`（增加约 `7.4 mm`）。因此不能仅凭 Success
或 Path Efficiency 说低摩擦“更好”；当前直接支持的结论是任务级成功保持，而停止稳定性
退化。25 个样本可提供配对的工程证据，但不足以单独做广泛 OOD 泛化或统计显著性声明。

## Delay 与 Motor full（同 25 个独立场景）

三项 full 都使用 `w04e010--w04e034`，并各自配对同一批 nominal EpisodeSpec。以下数值
不能跨条件汇总为单一 OOD 成功率，但可并列比较每个单因素的方向性：

| 条件 | Success | Median Completion Time | 相对 Nominal 配对中位差 | Median Path Efficiency | Median Stop Drift | 相对 Nominal 配对中位差 |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 25/25 | 12.58 s | — | 0.7422 | 0.0121 m | — |
| Low Friction 0.3 | 25/25 | 12.98 s | `+0.12 s` | 0.8685 | 0.0195 m | `+0.0084 m` |
| Delay 40 ms | 25/25 | 8.52 s | `-3.48 s` | 0.7522 | 0.0394 m | `+0.0275 m` |
| Motor 80% | 25/25 | 13.22 s | `+0.64 s` | 0.7341 | 0.0253 m | `+0.0141 m` |

Delay 40 ms 在全部 25 个配对场景中完成时间更短，但停止漂移在全部 25 个场景中更大，且
`wz` 饱和比例的配对中位差为 `+0.0520`。因此更快到达不等于控制质量更高：它的到达后
稳定性退化最明显。Motor 80% 在全部 25 个场景中完成更慢、停止漂移更大；路径效率也在
22/25 个配对场景中低于 nominal。Low Friction 的轨迹更短、路径效率更高，但停止漂移在
23/25 个场景中更大。

## Low Friction 修复与产物边界

早期输出目录 `artifacts/week04/02_classical_pointgoal_ood/model_1500_constrained/quick/`
中的 Low Friction 结果不纳入本报告：当时只修改了双足，而 floor 仍是 `1.0`；MuJoCo
同优先级接触使用较大的摩擦值，因此实际接触仍为 `1.0`，轨迹与 nominal 完全一致。

修复后运行时同时设置并读回 floor 与双足的 `0.3`，原始来源为：

```text
artifacts/week04/02_classical_pointgoal_ood/model_1500_constrained_friction_readback/
├── quick/summary/classical_pointgoal_ood_quick_model_1500.json
└── full/summary/classical_pointgoal_ood_full_model_1500.json
```

对应 source summary SHA-256 分别为
`7f106064dc27553ca7c6f0a08431a98854818ed75b81e1f37bb932a62d9b74a6` 与
`ae1dd05e3eabe626bcb4e2e933ba3bdcfdad96854e8ef2c16a9f4d4f1a651a4a`。

Delay 40 ms full 的原始 summary 位于
`artifacts/week04/02_classical_pointgoal_ood/model_1500_constrained_delay40_full/full/summary/`
（SHA-256：`56061a0148848f4e934f2e873cc501d7fedb2674b89a1e7ca28fad21f24039cb`）；Motor
80% full 位于
`artifacts/week04/02_classical_pointgoal_ood/model_1500_constrained_motor80_full/full/summary/`
（SHA-256：`f10d6310329cc82d3f25248f272522f22c7ff018e1b42f6bb3342bce764d4ad4`）。

协议字段目前仍名为 `foot_friction`，但本报告以 full summary 的
`runtime_perturbation.applied_friction` 为准，实际范围是 floor 加双足。字段命名的后续
整理不得篡改本次已冻结、已完成的 v1 证据。

## 可复现与下一步

静态检查：

```bash
cd ~/projects/microduck-embodied
python3 evaluation/classical_pointgoal_ood.py --validate-only
```

Low Friction viewer（不计入 quick/full）：

```bash
cd ~/projects/microduck_rl

uv run python \
  ~/projects/microduck-embodied/evaluation/classical_pointgoal_ood.py \
  --viewer \
  --condition low_friction_0p3 \
  --episode-id w04e010 \
  --output-dir ~/projects/microduck-embodied/artifacts/week04/viewer_comparison
```

三项计划内轻量 OOD 条件均已完成。下一步应整理配对轨迹、成功/失败案例和 viewer 对照，
而不是在同一冻结 v1 协议中根据这些结果继续追调 Controller 参数。

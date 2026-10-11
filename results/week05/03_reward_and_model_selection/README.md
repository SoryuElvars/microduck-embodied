# Week 05 / 03：近目标减速 Reward、课程学习与旧低层上的 K10 选模

在不改变动作范围、K、低层 ONNX 或 Week04 benchmark 的前提下，训练加入 near-goal speed shaping：成功圈附近的目标速度为 `0.08 m/s`，远处 cruise speed 为 `0.20 m/s`，近目标距离为 `0.35 m`。0.08 与 0.09 m/s 快筛差异有限，选择 0.08 m/s。K10 在 5 Hz 决策频率下训练，目标采样采用从前方近距离逐步扩展到全方向、1.8 m 的课程学习。

正式 checkpoint sweep 使用冻结 200 EpisodeSpec 与旧低层 `model_1500.onnx`：

| checkpoint | 位置成功 | 超时 | 成功后重离开 | 观察到停稳 |
|---|---:|---:|---:|---:|
| 600 | 200/200 | 0 | 1 | 200/200 |
| 900 | 192/200 | 8 | 54 | 192/200 |
| **1200** | **200/200** | **0** | **0** | **200/200** |
| 1499 | 199/200 | 1 | 9 | 199/200 |

因此冻结历史高层候选为 `model_1200.onnx`，K=10、5 Hz；其 SHA、命令约束、旧低层 SHA 和正式证据均在 [selected_model.json](selected_model.json) 中记录。900/1499 说明 PPO 后续更新并不单调变好，不能仅按最终 iteration 选模型。

这个选择只对旧 `model_1500` 的命令响应有效。若采用新的低层 `model_1498`，本报告的高层候选必须重训与重选。

原始证据按实验阶段保留在：

- `artifacts/week05/07_near-speed_k10_quick_eval/`：初版 near-goal speed quick；
- `artifacts/week05/08_near-speed009_k10_quick_eval/`：`0.09 m/s` 对照；
- `artifacts/week05/09_near-speed008_k10_formal_eval/`：进入 checkpoint sweep 前的 `0.08 m/s`
  诊断；目录名虽含 `formal`，当前保留的 summary 仍是 smoke/quick；
- `artifacts/week05/10_checkpoint_selection_k10/`：600/900/1200/1499 的 200-Episode formal 与重计分。

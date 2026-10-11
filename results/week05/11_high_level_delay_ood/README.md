# Week 05：冻结 K10 Navigator 的高层命令延迟 OOD

## 问题边界

本实验只延迟高层 RL Navigator 输出的 `VelocityCommand` 到冻结 locomotion policy 接收命令
之间的通道。它不是 Week 03 的低层 actuator/joint-action delay：机器人动力学、低层
ONNX、目标和 RobotState 真值指标均保持不变。

- 高层策略：冻结 K10 `model_1200.onnx`，SHA-256 `554f420964daa7811ca09e1555d9711ffdf4df5a99318056db2cd987448c3642`
- 低层策略：冻结 `model_1500.onnx`，SHA-256 `533b820c13b3782f2c35bfaa7171c29b095d1b5c121c0fe5d8bef80ab37e785d`
- 高层频率：5 Hz；低层控制周期：20 ms
- OOD 协议：[`week05_rl_high_delay_ood_v1.json`](../../evaluation/pointgoal_protocols/week05_rl_high_delay_ood_v1.json)，SHA-256 `c5eeef0902949588643908e2e3071dcce9e1099939375159803d1a4c18722040`
- 基础清单：冻结 Week04 200 EpisodeSpec；quick 使用前 10 个固定 EpisodeSpec。

每个控制步写入 FIFO delivery trace，包含生成/应用命令的序号与时刻。只有实际 readback
严格等于配置的 0、20、40 ms 才接受该 episode。

## 已完成：静态校验、smoke 与 quick

smoke 对三个延迟档各运行一个 5 秒短时 episode；因短时限 timeout 是预期现象，不作能力
结论。三个档位均通过 readback：0 ms 为 0 步、20 ms 为 1 个低层步、40 ms 为 2 个低层步。

quick 使用同一冻结 10 EpisodeSpec：

| 高层命令延迟 | 成功 | 跌倒 | 超时 | 中位完成时间 (s) | 中位路径效率 | 中位最终距离 (m) | 到达后重离开 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 ms | 10/10 | 0 | 0 | 14.59 | 0.696 | 0.187 | 0/10 | 通过 |
| 20 ms | 10/10 | 0 | 0 | 13.63 | 0.710 | 0.186 | 0/10 | 通过 |
| 40 ms | 10/10 | 0 | 0 | 14.32 | 0.714 | 0.188 | 0/10 | 通过 |

quick 的硬 Gate 为成功率 100%、Fall Rate 0、invalid-state rate 0，三个条件均通过。
20/40 ms 的少量完成时间和效率变化不构成“延迟改善策略”的结论：样本只有 10 集，必须由
同一 200 集正式协议确认。

原始 quick summary：
`artifacts/week05/11_high_level_delay_ood_k10/quick/summary.json`，SHA-256
`a4e4edd5874b5e37e6893cc9eccf0e8936ad4ca99098d21268c5fe8a9546929b`。

## 下一步：用户启动 formal

quick 已通过。正式评测将按三档延迟各运行完整的 200 EpisodeSpec，共 600 个 episode，
写入独立的 `full/` 目录并支持 progress 断点续跑。运行命令：

```bash
cd /home/elvars/projects/microduck-embodied

/home/elvars/projects/microduck_rl/.venv/bin/python \
  evaluation/rl_pointgoal_high_delay_ood.py \
  --full
```

formal 结果用于决定：高层延迟在 0--40 ms 内是否确实保持安全，以及是否需要进入高层
domain-randomized Navigator 训练。它不替代已计划的低层延迟 domain randomization。

# Week 05：速度感知的到达条件与旧模型诊断复评

## 结论与边界

训练侧的 PointGoal 成功条件已由“进入 `0.20 m` 圈并保持 `0.5 s`”改为：
处于成功圈内，最近 `0.5 s` 高层采样的**平面速率均值**不超过 `0.10 m/s`，
**有符号角速度均值的绝对值**不超过 `0.20 rad/s`，且这个稳定状态再连续保持
`0.5 s`。角速度先作有符号平均，以免底层步态的左右摆动被误判为持续转向；
平面速率逐样本取模，不允许前后移动互相抵消。没有加入最终朝向目标，也没有
改 Reward 权重、动作边界、低层模型或 Week04 冻结 benchmark。

现有 K2/K5/K10 `model_99` 的**旧轨迹短窗口**重计分，各为 **0/10 观察到停稳
到达**；同一方法重计分 Week04 Classical Constrained 的 10 条 quick 轨迹，
在三种高层采样间隔下均为 **10/10**。旧轨迹在原位置成功后仅继续记录约 2 秒，
所以这里的 `0/10` 不是新规则下的完整 Episode 成功率。

随后在相同冻结场景上执行新的 **live quick**：旧 ONNX 继续控制到新停稳条件达成
或 20 秒导航 timeout，不因原位置条件提前结束；达成后继续观测 2 秒。结果如下。
旧模型权重未改变，本次没有重新训练，也没有启动正式评测。

| 旧模型 | 原 Week04 位置到达 | 短轨迹重计分 | 新条件 live quick | live 跌倒/超时 |
|---|---:|---:|---:|---:|
| RL K2 | 10/10 | 0/10 | **10/10** | 0/0 |
| RL K5 | 10/10 | 0/10 | **0/10** | 0/10 |
| RL K10 | 10/10 | 0/10 | **3/10** | 0/7 |
| Classical Constrained（同场景；仅用于短轨迹阈值可行性对照） | 10/10 | 10/10 | 未重跑 | — |

K2 live 的停稳完成时间中位数为 `15.0 s`，旧位置条件的中位数约 `12.52 s`。
达成后继续运行 2 秒，K2 的到达后平面速度中位数约 `0.069 m/s`，10 次均未重新
离开成功圈；K10 的 3 次成功到达后速度中位数约 `0.086 m/s`，亦未离圈。
这表明 K2 旧策略可以在更长的持续控制后停稳，但不能据 10 个 quick 场景就冻结
K 或宣称 OOD/实机能力。K5、K10 的 timeout 多数最终仍接近目标，主要缺口是
无法持续满足位置与低速的联合条件，而不是本轮发生跌倒。K10 的 3 次成功不应只
用成功子集时间与 K2 全部 10 次比较。

## 阈值依据

来源是本地冻结轨迹
`artifacts/week04/01_classical_pointgoal_benchmark/model_1500/quick/constrained/raw/`
的 10 个 Nominal quick Episode。仅使用旧到达后 `cmd_vx=cmd_wz=0` 的站立片段，
按 K2/K5/K10 对应的 `40/100/200 ms` 间隔取样，再计算覆盖至少 `0.5 s` 的窗口。
平面速率是 `mean(sqrt(actual_vx²+actual_vy²))`；角速度是
`abs(mean(actual_wz))`。各采样间隔的 99 分位结果如下：

| 模拟高层 K | 窗口样本数 | 零命令窗口平面速率 P99 | 零命令窗口角速度 P99 |
|---|---:|---:|---:|
| 2 | 13 | 0.078 m/s | 0.067 rad/s |
| 5 | 5 | 0.084 m/s | 0.146 rad/s |
| 10 | 3 | 0.084 m/s | 0.181 rad/s |

初版统一阈值 `0.10 / 0.20` 略高于这批站立基线的最大 P99，且显著低于 RL
到达后仍在运动的观测。它是**仿真代理阈值**，不是实机静止标准；10 个场景的
尾部分位也不能作为统计保证。没有选单帧角速度阈值，因为零命令站立片段的瞬时
角速度 P95 仍约 `0.52 rad/s`，主要体现步态摆动。实机或 OOD 要重新校准传感器
噪声和状态估计，不得把本阈值直接视为硬件验收值。

## 复现与验证

训练逻辑位于官方仓库 `src/mjlab_microduck/tasks/mdp.py` 的
`PointGoalCommand.update_success`，数值写在
`src/mjlab_microduck/tasks/microduck_pointgoal_navigation_env_cfg.py`。
三组训练配置共用同一物理时间窗口和阈值；K2/K5/K10 对应 `13/5/3` 个高层
观测点。每个 Episode 重采样时清零窗口历史。相关官方 CPU 单测 `15 passed`。

旧模型重计分使用本仓库 `evaluation/pointgoal_stable_arrival_rescore.py`，只读取原始
CSV，输入与输出哈希记录在本地
`artifacts/week05/03_stable_arrival_rescore/k{2,5,10}/{smoke,quick}/summary.json`。
每组均按 `--validate-only → --smoke → --quick` 完成。示例命令：

```bash
cd /home/elvars/projects/microduck-embodied
python3 -m evaluation.pointgoal_stable_arrival_rescore \
  --raw-dir artifacts/week05/02_rl_quick/k5_frozen_eval/quick/raw \
  --k 5 --max-planar-speed-mps 0.1 --max-abs-yaw-rate-radps 0.2 \
  --output-dir artifacts/week05/03_stable_arrival_rescore/k5 --quick
```

live quick 使用独立的 `evaluation/rl_pointgoal_stable_quick.py`，不改 Week04
`run_episode`。每组同样通过 `--validate-only → --smoke → --quick`，原始 CSV 与
含模型、协议、代码哈希的摘要保存在本地
`artifacts/week05/04_stable_live/k{2,5,10}/{smoke,quick}/`。示例命令从官方仓库运行：

```bash
cd /home/elvars/projects/microduck_rl
uv run python /home/elvars/projects/microduck-embodied/evaluation/rl_pointgoal_stable_quick.py \
  --high-policy logs/rsl_rl/pointgoal_navigation_k5/2026-09-28_22-39-23_week05-nominal-k5-seed42/2026-09-28_22-39-23_week05-nominal-k5-seed42.onnx \
  --k 5 --output-dir /home/elvars/projects/microduck-embodied/artifacts/week05/04_stable_live/k5 \
  --quick
```

将 live CSV 再交给离线重计分逻辑，30 个 Episode 的达成标志与首次达成时间全部
一致。将 live 与原冻结 quick 的前缀逐步配对比较，K2/K5/K10 分别有
`6864/6692/6620` 个共同 warmup/navigation 步，时间、命令、实际速度与世界坐标
的最大绝对差均为 `0`；差异只从旧位置成功后的不同终止条件开始。官方 CPU 单测
`15 passed`，个人仓库单测 `108 tests OK`，两个仓库的
`git diff --check` 均通过。下一步若要验证新成功条件能否**教会**策略停车，需要在
此单变量配置下重新训练；旧 ONNX 的 live 结果不能当成重训收益。训练前先做官方
64-env 短程 smoke，正式训练由用户启动。

# Week 05 RL Navigator 训练框架（scaffold）

## 当前结论

Week 05 的第一版分层训练链路已经搭通，并已完成 `K=2 / 5 / 10` 各自独立的
`64 envs × 5 iterations` CUDA 冒烟训练。三组均可完成采样、PPO 更新、checkpoint 保存
和 ONNX 导出，未观察到 `fall` 或 `nan_state`。这些 checkpoint 只是链路验证产物，**还不
是可用的高层 Navigator 模型**；当前证据不能支持成功率、收敛、K 值优选或泛化结论。

```text
High-level PPO: 7-D observation -> 2-D normalized action
        ↓ clamp + rate limit
VelocityCommand: [vx, 0, wz]
        ↓ held for K low-level steps
Frozen model_1500 TorchScript: 61-D observation -> 14 joint actions
        ↓
MuJoCo + BAM
```

训练代码位于官方仓库 `/home/elvars/projects/microduck_rl`：

- `src/mjlab_microduck/tasks/mdp.py`（文件末尾的 Week-05 PointGoal 分组）
- `src/mjlab_microduck/tasks/microduck_pointgoal_navigation_env_cfg.py`
- `tests/test_pointgoal_navigation_cfg.py`

与 Week 4 `Navigator` 接口一致的运行时壳位于本仓库：

- `navigation/rl_navigator.py`
- `tests/test_rl_navigator.py`

## 已验证

- 高层 actor/critic observation 均为 7 维：
  `[dx_body, dy_body, vx_body, vy_body, wz_body, previous_cmd_vx,
  previous_cmd_wz]`；
- 高层 action 为 2 维，映射到 `vx ∈ [0, 0.25] m/s`、
  `wz ∈ [-0.5, 0.5] rad/s`，`vy=0`；
- 线加速度限制 `0.5 m/s²`，角加速度限制 `1.0 rad/s²`；
- 没有注入 Classical Navigator 的近目标减速或停车启发式；
- 冻结底层 observation 为 61 维，输出为 14 维；训练侧使用一次 batched
  TorchScript 推理，避免逐环境 CPU ONNXRuntime；
- `K=2 / 5 / 10` 的控制周期分别为 `0.04 / 0.10 / 0.20 s`，三者均先完成
  1-env reset + step，再完成 `64 envs × 5 iterations` CUDA 冒烟；
- 三个 K 的 PPO rollout 物理时长都为 `2.4 s/iteration`，折扣因子按物理时间换算，
  避免 K 之外的数据量与 discount horizon 混杂；
- 个人仓库新增 6 个 RL Navigator 单测，全仓共 105 个单测通过；官方仓库相关配置
  测试按 pytest 参数展开后 12 个通过；两个仓库 `git diff --check` 均通过。

## 64-env × 5-iteration 冒烟记录

统一条件：`seed=42`、CUDA、TensorBoard logger、每个 K 的 rollout 物理时长均为
`2.4 s/iteration`。表中数值仅用于检查有限值、终止项和数据流，不作为 benchmark
成功率，也不能据此选择 K。RSL-RL 日志里的 termination 数值是跨 rollout 聚合项，可能
大于 1，不能直接解释为 Episode 比例。

| K | 高层周期 | 最后一次 Mean reward | final distance | fall | nan | 结果目录 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | `0.04 s` | `4.19` | `0.2938 m` | `0` | `0` | `logs/rsl_rl/pointgoal_navigation_k2/2026-09-28_22-11-09_week05-nominal-k2-seed42` |
| 5 | `0.10 s` | `5.37` | `0.1987 m` | `0` | `0` | `logs/rsl_rl/pointgoal_navigation_k5/2026-09-28_22-22-14_week05-nominal-k5-seed42` |
| 10 | `0.20 s` | `5.53` | `0.2267 m` | `0` | `0` | `logs/rsl_rl/pointgoal_navigation_k10/2026-09-28_22-22-55_week05-nominal-k10-seed42` |

首次 K2 保存暴露了高层 action 名称与原有低层 ONNX metadata 假设不兼容的问题；修复
metadata 适配及实际 runner 绑定后，使用 K2 的 `model_4.pt` 独立复验导出成功，随后 K5、
K10 在训练保存阶段直接生成 ONNX。该问题属于导出基础设施，不改变 Reward 或 Controller。

## 当前草案参数（训练前必须评审）

下列值已经进入第一版配置，目的是让环境可构建和 smoke；它们还不是经过训练证据选择的
最终值：

| 项目 | 当前草案 |
| --- | ---: |
| progress weight | `+5.0` |
| one-shot goal weight | `+10.0` |
| linear acceleration cost weight | `-0.2` |
| yaw acceleration cost weight | `-0.05` |
| fall weight | `-10.0` |
| timeout remaining-distance weight | `-2.0` |
| Stage 1 -> 2 | iteration `500` |
| Stage 2 -> 3 | iteration `1000` |
| Stage 1 primary/tail | `90% / 10%`，primary `±45°` |
| Stage 2 primary/tail | `80% / 20%`，primary `±90°` |
| episode timeout | `20 s` |

这些值已经作为本轮 smoke 草案保持不变。下一步若进入 quick screen，应继续对
`K=2 / 5 / 10` 使用 `64 envs × 100 iterations`、同一 `seed=42` 和每 25 iterations
保存 checkpoint 的相同预算，并全部上传到 W&B 项目 `mjlab_microduck`。三组不合并
Return 或成功率；quick 前不得根据上述 5-iteration 数值临时调整 Reward。若 100
iterations 后差异不明确，三组一起延长到 200 iterations。

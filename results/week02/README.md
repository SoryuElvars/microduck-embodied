# Week 02：Locomotion 自动评测与训练诊断

本周结果按“研究问题”组织。这里仅保留适合审阅和提交 Git 的报告、图表与关键
汇总；逐步 CSV、逐命令 JSON 和 checkpoint 中间产物统一保存在本地
`artifacts/week02/`，不提交 Git。

| 序号 | 实验 | 核心问题 | 主要结论 | 状态 |
| --- | --- | --- | --- | --- |
| 01 | [Official Locomotion Baseline Report](./01_baseline_5999/) | 八项指令、Return、典型轨迹与系统合约如何？ | 只有站立静止成功；运动跟踪存在死区、耦合和方向不对称 | 完成 |
| 02 | [Checkpoint 分析](./02_checkpoint_analysis/) | 缺陷在哪些 checkpoint 出现，是否随训练单调改善？ | 行为非单调；最终只形成明显正向转弯 | 完成 |
| 03 | [Action 镜像诊断](./03_action_symmetry/) | 不对称是否已经存在于策略映射内部？ | ONNX 动作映射存在明显镜像不等变 | 完成 |
| 04 | [Symmetry A/B 续训](./04_symmetry_ab/) | mirror loss 能否修复双向控制？ | 改善镜像误差，但没有修复负向转弯和前进偏航 | 完成 |
| 05 | [400-Episode 随机验证](./05_random_400/) | 随机组合速度指令下是否具备泛化控制能力？ | 0% 跌倒，但运动指令跟踪未达到项目门槛 | 完成 |

固定八指令矩阵、checkpoint 诊断和随机指令评测是不同协议，分别报告，不合并
Episode 数量或成功率。

## 目录约定

每项公开实验目录尽量保持相同结构：

```text
<experiment>/
├── README.md     # 主报告或实验索引
├── figures/      # 可审阅图表
└── summaries/    # 关键机器可读汇总
```

本地生成物使用与公开结果相同的实验编号：

```text
artifacts/week02/<experiment>/
```

这些文件可由 `evaluation/` 中的脚本重新生成，因此不进入版本控制。

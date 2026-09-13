# Week 03：底层模型选择与鲁棒性实验

本周先确定可用于 PointGoal Navigation 的底层 Locomotion Policy，再对优胜候选
运行统一的 Nominal/OOD 鲁棒性评测。大型 ONNX、逐步 CSV 和中间文件保存在
本地 `artifacts/week03/`，不提交 Git。

| 序号 | 实验 | 核心问题 | 当前结论 | 状态 |
| --- | --- | --- | --- | --- |
| 01 | [官方 `alpha_walking.onnx` 快筛](./01_official_alpha_walking/) | 官方推理模型能否直接作为导航底层？ | 接口无误；理想执行器和旧版精确 MJCF 仍暴露从静止启动门槛，不能直接冻结为通用底层 | [快筛与兼容性排查完成](./01_official_alpha_walking/compatibility_diagnosis.md) |

固定八指令、随机速度指令、鲁棒性参数扫描和 PointGoal 属于不同协议，分别报告，
不得合并 Episode 数或成功率。

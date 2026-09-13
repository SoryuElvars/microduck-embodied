# MicroDuck Embodied 协作约定

## 项目范围

- 本仓库 `/home/elvars/projects/microduck-embodied` 用于个人评测、实验结果、导航和展示。
- 官方训练仓库为 `/home/elvars/projects/microduck_rl`。不要把官方环境、模型或推理实现复制到本仓库。
- 开发环境是 Windows 主机上的 WSL2 Ubuntu 24.04；GPU 任务使用 WSL2 CUDA。

## 目录与产物

- `evaluation/`：评测、分析和绘图脚本。
- `tests/`：可在本仓库独立运行的测试。
- `results/`：适合审阅和提交 Git 的中文报告、图表和关键汇总。
- `artifacts/`：原始 CSV、中间文件和大型生成物，只在本地保留，不提交 Git。
- 结果报告默认用中文；代码标识符、字段名和必要的上游术语保留英文。

## 评测口径

- 始终记录 checkpoint、官方 commit、配置、随机种子、Episode 数和时长。
- 固定指令、随机组合指令和官方环境 Return 属于不同协议；不得默认合并 Episode 数或成功率。
- ONNX 部署形态默认没有训练 Reward。Episode Return 必须由官方 PT checkpoint + Reward Manager 评测获得。
- 当前 walking actor 合约为 61 维 Observation 和 14 维 Action；实施前仍需从当前上游代码核对。
- 训练完成、总 Reward 高或不跌倒都不足以证明策略合格；同时报告速度跟踪、方向对称性、Fall Rate 和成功门槛。
- 进行训练 A/B 时保留 baseline，一次只改一个因素，先做短程冒烟测试。

## 执行与验证

- 依赖 `mjlab`、官方环境或 GPU 的命令，从 `microduck_rl` 目录使用 `uv run ...`。
- 本仓库纯 Python 测试：`python3 -m unittest discover -s tests -v`。
- 完成修改后运行相关测试和 `git diff --check`；长时评测前先使用最小配置冒烟验证。
- 如果发现并验证了能稳定提高训练、评测或日常协作效率的方法，将可复用结论简洁补充到本文件；不记录未验证猜测和一次性技巧。
- 保留用户已有的未提交更改。不提交 checkpoint、ONNX、原始 CSV 或其他大型产物。
- 不自动 commit 或 push；将 Git 提交作为用户明确要求的独立步骤。

## 读取顺序

- 项目目标先看 `README.md`，周计划与验收项看 `MicroDuck_前六周实施计划.md`。
- 评测命令和指标边界看 `evaluation/README.md`，已完成结论从对应 `results/<week>/<experiment>/README.md` 读取。
- 只在任务相关时读取上述详细文件，不在每次任务中全量重读。

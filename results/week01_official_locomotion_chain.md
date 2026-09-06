# 第 1 周：官方 Locomotion 链路复现记录

记录日期：2026-09-06

## 结论

`Mjlab-Velocity-Flat-MicroDuck` 的工程链路已经完整跑通：

```text
环境与 CUDA 验证
  → PPO 训练
  → Checkpoint 保存与续训
  → ONNX 导出
  → ONNX Runtime 加载
  → CPU MuJoCo/BAM 推理
```

因此，第 1 周“掌握并跑通官方技术链路”的核心目标判定为完成。

但本次 `model_5999.pt` 只能作为一个**带已知缺陷的官方基线复现**，不能判定为合格的直线行走策略：在纯前进、零横移、零转向命令下，策略出现稳定的正向偏航，并且该问题在训练约 2000 iterations 后进入平台期，继续训练至 6000 iterations 仍未解决。

## 复现环境

| 项目 | 记录值 |
|---|---|
| 官方仓库 | `pollen-robotics/microduck_rl` |
| 官方 Commit | `29e887ecfbf5d37144759e5a9f8a176dfb83d547` |
| 任务 | `Mjlab-Velocity-Flat-MicroDuck` |
| 系统 | Windows + WSL2 Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Python 环境 | `uv` 管理的项目虚拟环境 |
| PyTorch | `2.9.1+cu128` |
| MuJoCo | `3.10.0` |
| ONNX Runtime | `1.24.4` |
| 并行环境数 | `4096` |
| 随机种子 | `42` |

第一个训练 Run：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 6000 \
  --agent.run-name baseline-flat-4096x6000
```

训练在 `model_5000.pt` 后中断，随后续训 1000 iterations：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run 2026-09-05_16-50-10_baseline-flat-4096x6000 \
  --agent.load-checkpoint model_5000.pt \
  --agent.max-iterations 1000 \
  --agent.run-name baseline-flat-resume-5000
```

## 产物

以下大文件保留在官方仓库的本地日志目录中，不提交到个人 Git 仓库：

```text
logs/rsl_rl/velocity/2026-09-05_22-08-33_baseline-flat-resume-5000/
├── model_5999.pt
├── microduck_velocity_flat_5999.onnx
└── 2026-09-05_22-08-33_baseline-flat-resume-5000.onnx
```

ONNX 静态检查结果：

| 检查项 | 结果 |
|---|---|
| ONNX Checker | 通过 |
| CPU Execution Provider | 加载与推理通过 |
| 输入 | `obs: float32[1, 61]` |
| 输出 | `actions: float32[1, 14]` |
| 零输入输出 | 全部为有限值，无 NaN/Inf |
| 显式导出与自动导出对比 | 100 组随机输入的最大绝对输出差为 `0.0` |

这说明当前偏航问题不是 ONNX 文件损坏，也不是两次导出结果不一致。

## W&B 曲线发现

`Metrics/twist/error_vel_yaw` 在约 1750～2000 iterations 后进入长期平台期。以下数值为 TensorBoard 原始标量在对应位置“最近 100 次记录”的均值：

| Iteration | `error_vel_yaw` |
|---:|---:|
| 1500 | 1.242 |
| 2000 | 1.135 |
| 3000 | 1.147 |
| 4000 | 1.095 |
| 5000 | 1.060 |
| 6000 | 1.077 |

该指标按控制步累计并按命令最大持续步数归一化，W&B 中的绝对值不能直接当成瞬时角速度；它的用途是比较训练趋势。曲线说明后 4000 iterations 没有继续改善偏航跟踪。

## 零转向直行复现

使用导出的 ONNX、默认平地场景、BAM M6 执行器模型和以下固定命令进行无窗口复现：

```text
vx = +0.30 m/s
vy =  0.00 m/s
wz =  0.00 rad/s
```

关键结果：

| 时间 | 实际前进速度 | 实际偏航角速度 | 累计航向 |
|---:|---:|---:|---:|
| 1 s | `+0.150 m/s` | `+0.142 rad/s` | `+8.1°` |
| 2 s | `+0.181 m/s` | `+0.250 rad/s` | `+23.3°` |
| 3 s | `+0.175 m/s` | `+0.418 rad/s` | `+46.6°` |
| 10 s | `+0.178 m/s` | `+0.371 rad/s` | `+166.5°` |

该现象是持续向同一侧画弧，而不是双足步态中围绕零点的小幅周期摆动。因此，本策略未通过直线行走质量验收。

## 官方链路中遇到的问题

### 1. WSLg 回退到 CPU 软件渲染

MuJoCo 初次推理只有个位数 FPS。OpenGL 探测结果为：

```text
RENDERER llvmpipe (LLVM 20.1.2, 256 bits)
```

设置以下环境变量后，渲染器正确切换到 RTX 4060，画面提升到约 30 FPS：

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

修复后探测结果：

```text
RENDERER D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)
```

两个变量已加入 WSL 用户的 `~/.bashrc`。

### 2. 推理脚本只监听终端键盘

`scripts/infer_policy.py` 从启动它的终端读取按键，不从 MuJoCo Viewer 窗口读取。正确操作是把焦点切回终端后直接按键，不需要回车。如果终端显示 `stdin is not a TTY`，该启动方式无法使用键盘控制。

### 3. 训练完成不等于策略质量合格

本次训练的总 Reward、Episode Length、Checkpoint 保存和 ONNX 推理均正常，但纯前进命令仍产生严重偏航。工程成功与策略质量必须分别验收，不能仅凭 `6000 iterations` 或总 Reward 宣称任务成功。

## 第 1 周验收判定

| 门禁 | 状态 | 说明 |
|---|---|---|
| 环境、依赖与 CUDA | 通过 | WSL2 CUDA 可用 |
| PPO 训练与续训 | 通过 | 完成至 iteration 5999 |
| Checkpoint | 通过 | 可加载并由官方导出器使用 |
| ONNX Export | 通过 | Normalizer 随官方导出链路写入模型 |
| ONNX Runtime | 通过 | 61 维观测到 14 维动作 |
| MuJoCo 部署形态推理 | 通过 | 可运行、可接受终端速度指令 |
| 直线行走质量 | **未通过** | 10 秒累计偏航约 `166.5°` |
| Walking Video | 未完成 | 待获得有代表性的可展示策略后录制 |

最终状态：**第 1 周工程链路完成，策略质量带缺陷；允许进入第 2 周建立定量 Benchmark，但当前模型只能作为失败基线，不能作为冻结的合格底层策略。**

## 下一步

第 2 周首先将本次人工直行检查固化为自动评测，再决定第二轮训练配置。优先验证：

1. 固定命令下的前进速度误差、平均偏航角速度与 10 秒净航向变化；
2. 左右对称性增强是否能消除单侧偏置；
3. 收紧或提高角速度跟踪奖励是否能突破当前平台；
4. 是否需要加入绝对航向保持，而不只跟踪瞬时偏航角速度；
5. 每隔固定 Checkpoint 自动运行短时验收，避免长时间训练后才发现方向偏置。

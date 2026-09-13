# 官方 `alpha_walking.onnx` 兼容性排查

日期：2026-09-13

## 结论

没有发现 Observation 排列、Normalizer、关节顺序或 ONNX Runtime 前向计算错误。
把模型切换到官方浏览器演示所用的旧版 MJCF、理想位置执行器和 MuJoCo 3.11 后，
从静止直接给正常幅值的后退、横移和负向转动仍然基本不响应。因此，原快筛失败
不能归因于本项目接错模型接口。

目前确认了两个叠加现象：

1. 当前 BAM M6 复现路径会进一步削弱这份旧策略的运动响应；
2. 策略存在明显的从静止启动门槛和状态依赖。先以 `vx=+0.25 m/s` 前进约 1 秒，
   进入周期步态后再切换，后退、双向横移和双向转动才会出现。

这个“前进预启动”可以让 MuJoCo Viewer 中的键盘演示基本可操作，但它不是合格的
底层速度跟踪策略修复：正常幅值下仍显著欠跟踪，而且从静止不能对任意方向立即响应。
按项目的固定八指令门槛，官方模型仍不通过，不能直接冻结为最终导航底层。

## 排查结果

| 项目 | 证据 | 判断 |
| --- | --- | --- |
| ONNX I/O | `obs: float32[1,61] → actions: float32[1,14]` | 一致 |
| Observation | `gyro(3) + projected_gravity(3) + joint_pos(14) + joint_vel(14) + last_action(14) + command(13)` | 与官方 Runtime 合同一致 |
| Command | `twist(3) + head_pose(4) + body_pose(6)`，未使用槽为零 | 一致 |
| Joint order | 14 个关节名及顺序与 ONNX metadata 一致 | 一致 |
| Default pose | 当前推理常量与 ONNX `default_joint_pos` 一致 | 一致 |
| Normalizer | ONNX 图中包含 61 维均值和除数，图首为 `Sub → Div` | 已内嵌，不应外部再归一化 |
| ONNX Runtime | 本地 ORT 1.24.4 与 ONNX ReferenceEvaluator 最大绝对差 `1.31e-6` | 前向计算一致，版本差不足以解释方向失效 |

## Runtime 与模型文件复核

- Policy revision：`088524a64e2557dc453256b6071dbb9d23888802`
- `alpha_walking.onnx` SHA-256：
  `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`
- 官方浏览器模拟器 revision：
  `023172c8a7d629b5258d90364c13bafe013abbfa`
- 本地测试 ONNX Runtime：`1.24.4`
- 官方浏览器演示声明的 ONNX Runtime Web：`^1.27.0`
- 本地默认 MuJoCo：`3.10.0`
- 对齐浏览器版本复核：MuJoCo `3.11.0`
- ORT 输出与 ONNX ReferenceEvaluator 在随机输入上的最大绝对差：
  `1.31e-6`，不足以解释运动方向失效。

官方浏览器演示使用的 `BEST_alpha_walking.onnx` 与本项目下载的
`alpha_walking.onnx` SHA-256 完全相同。它使用旧版 `robot_allcollisions.xml`、
理想位置执行器、`action_scale=1.0`、`0.005 s` timestep 和 decimation 4。
本项目已将该演示的 MJCF 与 mesh 固定到本地忽略目录，专用于兼容性复核：

```text
artifacts/week03/01_official_alpha_walking/runtime_reference/
```

当前 `microduck_rl` checkout 的三个 MJCF 哈希都与这份旧模型不同。使用旧版精确
MJCF 后仍复现相同的方向性问题；换用 MuJoCo 3.11 也没有质变。因此 MJCF 和
MuJoCo 小版本差异不是主因。

## 执行器 A/B

原八指令 × 5 seeds 快筛使用当前 CPU MuJoCo + BAM M6 nominal 配置：运动指令
`0/35` 通过。切到官方浏览器演示使用的理想位置执行器后：

- `vx=+0.25` 能稳定产生约 `+0.107 m/s` 前进；
- `wz=+1.0` 能从静止启动转动；
- 正常幅值的后退、双向横移和 `wz=-1.0` 仍几乎停在原地；
- 官方 `robotd` 默认的 `action_scale=0.9`、腿部低通 `0.7`、头部低通
  `0.5` 配合 BAM 并未恢复这些方向，反而进一步减弱了直接响应；当前
  `scripts/infer_policy.py` 本身默认仍是 `action_scale=1.0` 且不做目标低通。

所以 `--no-bam` 是官方浏览器演示的正确仿真 profile，但不是完整的策略修复。

## 从静止与运动中切换的差异

在旧版精确 MJCF + 理想位置执行器中，先前进 1 秒再切换到目标指令，5 秒测试段
得到：

| 目标指令 | 平均实际 `(vx, vy, wz)` | 解释 |
| --- | --- | --- |
| `vx=-0.20` | `(-0.085, +0.006, -0.126)` | 能后退，但明显欠跟踪并伴随转动 |
| `vy=+0.20` | `(+0.015, +0.041, -0.036)` | 有正确横移方向，但幅值仅约 20% |
| `vy=-0.20` | `(+0.016, -0.031, +0.032)` | 有正确横移方向，但幅值仅约 16% |
| `wz=-1.00` | `(+0.013, +0.002, -0.504)` | 负向转动恢复，约跟踪 50% |

`0.25–0.5 s` 的前进脉冲只足以激活负向转动；后退和横移约需 1 秒前进才稳定进入
运动分支。这说明故障主要是策略自身的吸引域/启动门槛，而不是简单的指令符号错误。

## 可用性判定

| 用途 | 判定 |
| --- | --- |
| Viewer 中观察官方模型能否运动 | 可用：理想执行器 + 初始前进 |
| 只用前进和双向转向的受限 PointGoal 原型 | 有条件可用：需预启动、保持运动状态并做闭环速度标定 |
| 固定八指令速度跟踪基准 | 不可用：从静止失败且欠跟踪严重 |
| Reality Gap / BAM 鲁棒性优胜模型 | 不可用：BAM 下响应进一步减弱 |
| 最终冻结的通用导航底层 | 不可用：应继续训练自己的候选模型 |

## Viewer 临时可用配置

以下命令使用已固定的官方浏览器演示 MJCF、理想位置执行器，并从启动时直接给
`vx=+0.25`，避免先落入静止吸引域：

```bash
cd ~/projects/microduck_rl

uv run python scripts/infer_policy.py \
  --scene ~/projects/microduck-embodied/artifacts/week03/01_official_alpha_walking/runtime_reference/app/public/robot/mjlab/scene_reference.xml \
  --walking ~/projects/microduck-embodied/artifacts/week03/01_official_alpha_walking/model/alpha_walking.onnx \
  --standing ~/projects/microduck-embodied/artifacts/week03/01_official_alpha_walking/model/alpha_stand.onnx \
  --new-cmd-obs \
  --no-bam \
  --lin-vel-x 0.25
```

操作时先让机器人持续前进约 1 秒，再切换后退、横移或负向转动。按空格完全停下
以后，若目标方向重新不响应，再用前进键激活约 1 秒。这个流程只用于人工观察和
受限导航原型，不应改变固定八指令从静止评测的口径。

## 官方依据

- [官方 Policy manifest 合同](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md)
- [官方 robotd Observation 与控制路径说明](https://github.com/pollen-robotics/microduck/blob/main/docs/design/robotd-design.md)
- [官方浏览器 MuJoCo 模拟器](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
- [官方模型不可变来源信息仍缺失的公开 Issue](https://github.com/pollen-robotics/microduck_rl/issues/40)

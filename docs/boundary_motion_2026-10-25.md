# 边界运动一致性修复与配对验证（2026-10-25）

本阶段修复了边界截断后的虚假速度、故障恢复后的遗留速度，并接入可选只读
轨迹记录。最终 720 个配对回合显示速度错配消除、碰撞和奖励有所改善，但学习
策略成功率仍为 0，贴边停留没有消失。本轮没有重新训练或更换奖励、网络结构。

## 问题与修复

原实现先计算指令速度，再把新位置截断到地图范围，却仍把未执行的指令速度
写入物理状态。例如节点在右边界继续向右发出 3 m/s 指令，实际位移为零，但
自身观测、邻居通信及速度一致性指标仍认为其在向右运动。

现在保留指令速度用于控制努力量计算，状态速度采用
`(本步结束位置 - 本步开始位置) / dt`。它是时间步内的实际平均运动速度：

- 完全被边界阻挡时为零，部分越界时反映实际走过的距离。
- 沿墙运动保留切向分量，朝向跟随实际运动；完全静止时保留原朝向。
- 能耗继续按原指令速度范数计算，贴墙发出指令不会被误当作零控制消耗。
- 无动作的离线节点本步速度归零，步末恢复时不会继续携带离线前的速度。
  故障开始步已经执行的运动仍保留，不被后来的故障抹去。

最后一项是在独立复核中进一步复现的既存问题。真实两步例子：节点先移动
0.5 m 后离线，下一步未动而恢复，原来自身观测仍带着 1 m/s；现已变为 0。

## 轨迹与指标

`run_episode` 新增可选 `trajectory` 列表，默认不记录。记录只发生在执行后，
不进入策略输入，也不改变回合指标。每步保留位置、原动作、指令速度、位移
平均速度、状态速度、在线状态、朝向、能量、边界截断、最小在线节点距离、
碰撞计数、发现率、拓扑版本和指令速度差异。

速度错配率检查**所有步末在线节点**，包括本步未收到动作的恢复节点，容差为
1e-8；离线节点不作为可用状态样本。边界占比的分母则是实际收到动作的节点
时间步，包含 NOOP，两项指标分母不同。不存在可检查样本或事件时保持空值。

新入口 `scripts/evaluate_motion.py` 对全部回合统计；默认仅保存各场景、策略
第一个回合的完整轨迹，控制磁盘占用。提供检查点时画学习策略的轨迹、最小
距离、贴边比例和指令速度差异图，规则及补位轨迹仍保存为 JSONL。代表回合
由回合编号预先确定，可能没有碰撞；最终统计始终使用全部 20 个回合。

## 环境版本与历史结果

当前运动语义为 `displacement_velocity_v2`，写入训练配置和评估来源清单。
历史文件缺少字段时按 `command_velocity_v1` 解释，不回写旧数据。

- 消融汇总和同环境试训分析拒绝混合不同运动语义的训练与评估。
- 当前环境续跑旧运动训练时，不复用旧阶段；保留归档后重新运行，建议新目录。
- 诊断入口默认拒绝运动语义变化；显式 `--allow-motion-shift` 才允许旧模型在
  新环境中诊断，同时记录训练版本、当前版本及变化标记。
- 独立运动评估允许同一旧模型分别在两个环境中执行，这正是本轮对照；它不
  被标记为同环境训练复现。旧模型的维度仍兼容，但速度观测含义已经改变。

## 实验协议

两个固定检查点均来自之前训练种子 0、25,600 环境步的 Huber 试训：

- 无记忆：`outputs/training_stability_huber_20261024/training/mappo_checkpoint.pt`。
- 有记忆：`outputs/coverage_memory_pilot_20261025/training/mappo_checkpoint.pt`。

每个模型、每种运动语义都运行规则、补位、学习三个策略；每策略运行正常、
链路/节点故障和通信扰动三个场景，每场景 20 个回合。测试种子预先固定为
**50000–50019**，4 个节点、50×50 地图、6 个目标、dt=0.2、最多 300 步；
通信扰动组使用丢包 0.3、延迟 0.2、抖动 0.1、噪声标准差 0.01。
场景与模型字节指纹逐项核对，采用确定性策略，不根据结果选择模型或种子。

先在原环境保存 360 回合基线，随后进行 360 回合中间核查。复核发现恢复节点
统计盲点后，在独立旧环境副本和当前环境中使用**同一版最终记录器**，完整
重跑最终 720 回合；不覆盖前面的 720 回合。因此本阶段实际运行 1,440 个评估
回合，下面只报告最终同口径的 720 个。去重两个模型组重复的规则基线后，
最终包含 480 条不同策略/种子/环境记录，不作为独立训练种子重复。

旧环境副本来自提交 `57fcac2`，运动源码与修复前保存的原文件字节指纹一致；
只接入最终记录器，没有把当前运动代码伪装成旧环境。两种环境的实际四个
评估源码均保存精确快照并核验各自指纹。

## 真实结果

以下为每个场景 20 个回合的均值，箭头表示修复前到修复后。

| 模型 | 场景 | 累计奖励 | 碰撞计数 | 目标发现率 |
|---|---|---:|---:|---:|
| 无记忆 | 正常 | -33.225 → -21.978 | 32.50 → 21.20 | 30.00% → 30.00% |
| 无记忆 | 链路/节点故障 | -41.401 → -18.853 | 39.75 → 17.60 | 29.17% → 30.00% |
| 无记忆 | 通信扰动 | -33.081 → -21.959 | 31.85 → 21.20 | 29.17% → 30.00% |
| 有记忆 | 正常 | -63.228 → -58.910 | 67.25 → 62.85 | 30.00% → 30.00% |
| 有记忆 | 链路/节点故障 | -56.132 → -52.092 | 60.00 → 55.85 | 30.00% → 30.00% |
| 有记忆 | 通信扰动 | -63.573 → -59.254 | 67.60 → 63.20 | 30.00% → 30.00% |

所有学习组成功率仍为 **0/20**。规则组成功率为 65%、60%、65%，补位组均为
45%；规则及补位的奖励、发现率、碰撞等任务指标在两种环境下逐回合保持一致。
速度一致性误差按修正后的真实速度计算，其物理解释随版本改变，不把它与
任务成功率或协作质量混为一谈。

修复前无记忆模型的在线速度错配率约为 49.58%–51.10%，有记忆为
71.87%–72.58%；修复后所有策略、场景的步末在线样本均为 0，包括恢复节点。
无记忆模型贴边样本仍约 49.61%–51.19%，有记忆仍约 71.84%–72.52%，因此
修正速度并没有让模型学会离开边界。

本批学习回合中，没有首个碰撞早于首个贴边的情况。这只是时间顺序证据；
不同节点可能先后接触边界，不能据此认定每个碰撞都由边界错误造成。
碰撞仍采用每步接近的节点对累计计数，不是独立事故数。本轮只有一个训练
种子，没有重新训练，没有跨训练种子显著性或已收敛、泛化优势结论。

## 文件改动和中文说明

| 文件 | 完成内容 |
|---|---|
| `environments/continuous_2d.py` | 真实位移速度、静止朝向、恢复节点速度；保留指令能耗，解释时间步语义和版本 |
| `evaluation/evaluator.py` | 可选执行后轨迹记录，默认调用兼容 |
| `evaluation/trajectory.py` | 指令/位移/状态三种速度快照，边界和时序统计，轨迹图；注释解释分母及离线状态 |
| `scripts/evaluate_motion.py` | 全回合评估、选定轨迹、原始 CSV/JSONL、来源和代码指纹；拒绝覆盖 |
| `scripts/train_mappo.py` | 保存训练运动版本 |
| `scripts/evaluate_policies.py` | 保存常规评估运动版本 |
| `evaluation/ablation.py` | 训练版本规范化、拒绝不同运动语义的训练/评估混算 |
| `scripts/run_ablation.py` | 旧运动阶段保留归档再重跑 |
| `scripts/analyze_training_pilot.py` | 同环境试训分析的运动版本校验与报告 |
| `scripts/diagnose_training.py` | 默认拒绝运动变化，显式允许时标记跨环境诊断 |
| `tests/test_motion_consistency.py` | 13 项墙面、部分越界、切向、角点、能耗、通信、恢复、轨迹透明和真实 CLI 测试 |
| `tests/test_ablation_reporting.py` | 新增运动版本错配及坏配置类型拒绝测试 |
| `tests/test_ablation_resume.py` | 新增旧运动训练预览不复用且不改源结果测试 |
| `tests/test_training_pilot.py` | 新增试训评估运动语义错配测试 |
| `tests/test_training_diagnostics.py` | 新增显式跨运动版本诊断及标记测试 |
| `README.md` | 新入口、版本变化与旧模型处理说明 |
| `docs/training_diagnostics_2026-10-25.md` | 保留历史数据，补充当前命令与旧语义的复现边界 |
| 本文 | 协议、结果、文件说明及运行命令 |

合计 18 个文件，无新增依赖。重要新增函数均说明功能、参数、返回值和关键
逻辑；环境和指标处特别解释为什么区分指令速度、实际运动和离线状态。

## 验证和产物

- 边界和版本错配测试先失败再修复；独立审查又复现恢复状态问题和统计漏检，
  两项新增回归分别先失败、修复后通过，审查确认问题关闭。
- 最终完整回归 **217 项全部通过，113.653 秒**；之前 199 项，本轮新增 18 项。
- 最终四组共 76 个来源文件指纹匹配；36 份代表轨迹按最终函数重新汇总一致。
- 新口径旧环境的回合任务原始记录，与首次保存的旧环境基线字节完全相同。
- 已查看配对对照图及全部 12 张学习策略代表轨迹图，零成功率正常保留。
- 真实旧模型的 8 步跨环境诊断 CLI 执行成功，明确记录旧/新运动版本与变化标记。
- 原试训 23 项来源、前一轮 12 份诊断报告指纹未变；本轮 18 个文件语法、
  常见隐私模式和提交差异格式检查通过，实验产物确认为 Git 忽略项。

```text
outputs/boundary_motion_20261025/
├── baseline_source/                  # 最早保存的旧源码
├── before_{no_memory,memory}/        # 首轮基线，保留
├── after_{no_memory,memory}/         # 中间核查，保留
├── verified_before_source/           # 最终旧环境和最终记录器源码快照
├── verified_after_source/            # 最终新环境源码快照
├── verified_before_{no_memory,memory}/
├── verified_after_{no_memory,memory}/
│   ├── scenario_config.json
│   ├── episode_metrics.jsonl
│   ├── scenario_summary.csv/json
│   ├── motion_metrics.csv/jsonl
│   ├── evaluation_provenance.json
│   └── trajectories/*.jsonl,*.png
└── comparison/{motion_comparison.json,scenario_summary.csv,paired_deltas.csv,motion_comparison.png,...}
```

所有模型、日志、源码快照及图表保存在本地 `outputs/`，不随 Git 上传。

## 运行命令

在项目目录使用已有依赖的 Python，选择全新输出目录：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'

# 旧模型在当前修复环境中的运动评估；完整记录第一个回合。
python scripts/evaluate_motion.py --checkpoint outputs/coverage_memory_pilot_20261025/training/mappo_checkpoint.pt --episodes 20 --agent-counts 4 --max-steps 300 --seed 50000 --trajectory-episodes 1 --packet-loss-rate 0.3 --base-delay 0.2 --delay-jitter 0.1 --noise-std 0.01 --output-dir outputs/motion_reproduction

# 新环境的少量训练验证；本阶段正式结果没有使用新训练的模型。
python scripts/train_mappo.py --coverage-memory --value-loss-type huber --updates 1 --rollout-steps 8 --update-epochs 1 --agents 2 --max-steps 8 --hidden-dim 8 --output-dir outputs/motion_train_smoke

# 对旧模型做新环境诊断时，必须明确允许运动语义变化。
python scripts/diagnose_training.py --checkpoint outputs/coverage_memory_pilot_20261025/training/mappo_checkpoint.pt --run-config outputs/coverage_memory_pilot_20261025/training/run_config.json --allow-motion-shift --seed 30000 --rollout-steps 256 --output-dir outputs/motion_shift_diagnostic

python -m unittest tests.test_motion_consistency -v
python -m unittest discover -s tests -q
```

若完整复现旧/新配对，应在独立目录还原 `57fcac2` 的旧环境，接入本轮最终
`evaluation/evaluator.py`、`evaluation/trajectory.py` 和 `scripts/evaluate_motion.py`，
使用相同检查点、参数与种子分别执行；实际用到的精确文件已保存在本轮两个
`verified_*_source/` 快照中。不要在主工作区来回替换旧源码或覆盖已有实验。

## 仍需完成

学习模型仍长时间向边界运动且没有完成任务。下一步应在当前正确的运动环境
上制定单项策略对照，例如针对边界动作或局部碰撞的控制机制，并明确区分
规则控制层的贡献与模型学习的贡献；需要重新训练才能判断新环境下的学习效果。

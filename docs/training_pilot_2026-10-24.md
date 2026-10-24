# 单种子试训与独立评估：2026-10-24

## 结论与边界

本次沿用现有 Graph-MAPPO、注意力聚合、通信观测、奖励及故障机制，没有修改
核心算法，也没有降低任务难度。真实运行了 25,600 个环境步。

**本次没有得到有效协作策略。** 学习策略的独立测试成功率仍为 0，目标发现率
略有上升，但碰撞计数大幅增加，累计奖励下降。不能据此宣称训练收敛、鲁棒性
提升、注意力优于其他聚合器或具备跨规模泛化能力，也不能仅凭这个种子断言
该算法永远不可学习。

下一步应先诊断训练稳定性与运动退化，再决定修正方案；目前不建议直接扩大
到多种子、多个聚合器及更大智能体规模的正式矩阵。

## 固定协议与环境

- 训练：4 个智能体，50×50 地图，6 个未知目标，每回合最多 300 步，dt=0.2。
- 模型：attention 聚合、隐藏维度 64，共享 Actor、集中式 Critic。
- 优化：100 次更新，每次采样 256 步、4 个 PPO epoch；其余使用原有 MAPPOConfig。
- 随机性：模型初始化及训练起始种子为 0；85 个完成回合和最后一个未完成回合
  使用环境种子 0–85，检查点中的 next_reset_seed 为 86。
- 训练扰动：保留原先固定端点/时刻的链路与节点故障；通信参数均为 0。
- 独立测试：种子 10000–10019；每个场景、每种策略运行 20 个回合，规模仍为 4。
- 正常场景为理想通信；拓扑故障场景采用原有链路/节点故障；通信扰动场景采用
  丢包率 0.3、基础延迟 0.2、抖动 0.1、噪声标准差 0.01，不同时叠加拓扑故障。
- 学习策略评估沿用已有 deterministic=True；未训练模型与最终模型均不更新参数。
- Python 3.12.14、torch 2.13.0+cpu、numpy 2.5.2、matplotlib 3.11.1；没有新增依赖。
  本轮进程设置 OMP_NUM_THREADS=1、MKL_NUM_THREADS=1，不改变全局环境配置。

初始模型使用相同架构和种子单独保存，随后训练入口重新按种子 0 初始化。
原始与最终两次评估的场景协议完全一致，规则及规则补位基线逐回合结果也一致。
分析工具检查这些约束后，才对同一测试种子逐回合作差、再取均值。

规则策略可以读取成员索引/数量等观测元数据，学习模型只编码其定义的特征。
因此这里是不同系统策略的对照，不是严格同信息集的聚合器消融。

## 实际运行结果

训练完成了 85 个回合，成功数为 0；所有训练指标和模型参数均为有限数值。
初始到最终模型的参数 L2 变化约 4.79731，说明参数确实更新过，不是空训练。
前 10 次更新的平均每步团队奖励为 -0.070566，最后 10 次为 -0.716986。
后期 Value loss 明显增大，但这只能作为不稳定迹象，不能独立认定具体根因。

最初训练前后各执行 180 个评估回合，共 360 次执行。为补齐执行时的模型来源
指纹，另在全新目录各重跑 180 个回合；来源复核不增加独立样本数。
规则基线的重复结果去重后，
保留 240 个“模型/策略 × 场景 × 测试种子”结果；这不等于 240 个训练种子。
来源复核重跑的配置、逐回合结果和汇总表均与首轮逐字节一致，原先 8 个核心
源文件的 SHA-256 全部未变；加上复核重跑，评估总执行次数为 720。

| 场景 | 学习策略训练前→后累计奖励均值 | 训练后目标发现率 | 训练后学习成功率 | 规则成功率 | 规则补位成功率 |
|---|---:|---:|---:|---:|---:|
| 正常 | -1.276 → -80.061 | 25.83% | 0/20 | 11/20 | 10/20 |
| 链路/节点故障 | -2.774 → -67.043 | 26.67% | 0/20 | 9/20 | 8/20 |
| 通信扰动 | -1.208 → -70.580 | 26.67% | 0/20 | 11/20 | 8/20 |

学习策略的目标发现率分别从 22.50%、20.83%、22.50% 上升到上述水平，
但每回合平均碰撞计数分别从 0 增至 85.75、73.10、76.80，抵消了发现目标的收益。
这里的碰撞是每步近距离智能体对的累计计数，持续接近会重复记账，不是互不重复
的物理事故数量。图表不画跨训练种子误差棒，也不报告单种子的统计显著性。

实测用时：成本探测（额外 256 步）6.964 秒，主训练 148.878 秒，训练前评估
53.474 秒，训练后评估 59.736 秒。成本探测没有混入主试训日志或预算。

## 定位到的运动退化

额外重放正常场景的独立种子 10000，保留真实逐步位置、动作、奖励与碰撞。

| 策略/诊断模式 | 完成步数 | 目标发现率 | 碰撞计数 | 累计奖励 |
|---|---:|---:|---:|---:|
| 初始模型，确定性 | 300 | 2/6 | 0 | 14.500 |
| 最终模型，确定性 | 300 | 2/6 | 229 | -214.000 |
| 最终模型，随机采样 | 300 | 2/6 | 31 | -16.900 |
| 规则策略 | 173 | 6/6 | 0 | 74.837 |

最终确定性模型的四个节点在回合末均到达 x=0 的左边界，其中两个节点持续靠近。
该回合收益分解为：20（发现）−3（步数）−229（碰撞）−2（断连）=−214。
因此碰撞惩罚是本回合低收益的直接来源。源码也表明边界使用位置裁剪，不会
自动为学习策略提供反向折返控制；规则策略则有显式折返及邻居排斥。

随机采样对照只为诊断训练采样与确定性推理的差异，动作种子固定为 50000。
它仍未完成任务，且只重放一个回合，不能据此替换正式评估口径或声称问题已修复。

### 尚未确认的根因与后续顺序

1. 先进一步核查 PPO 概率比、团队优势/在线人数权重、截断回合自举及共享编码器
   的 Actor/Critic 梯度，区分实现缺陷与优化设置问题；本次没有擅自修正这些机制。
2. 比较训练采样和确定性推理的运动/碰撞分布；不能只把评估改成随机动作来掩盖
   部署策略退化。
3. 若要增加覆盖历史、角色信息、边界反馈或安全避碰约束，需先明确研究方案，
   一次只改变一个因素，保留本轮作为原方案对照；不直接改奖励后冒充同一实验。
4. 在单种子独立测试出现可解释改善后，再按预先固定的预算执行多种子实验。

## 产物与分析入口

原始配置、训练日志、初始/最终模型和前后逐回合结果保存在：

```text
outputs/training_pilot_20261024/
├── initial/mappo_checkpoint.pt
├── training/{run_config.json,training_metrics.jsonl,mappo_checkpoint.pt,training_curves.png}
├── evaluation_initial/{scenario_config.json,episode_metrics.jsonl,scenario_summary.csv,...}
├── evaluation_final/{scenario_config.json,episode_metrics.jsonl,scenario_summary.csv,...}
├── analysis/{pilot_analysis.json,paired_comparisons.csv,pilot_comparison.png}
├── evaluation_initial_verified/{evaluation_provenance.json,episode_metrics.jsonl,...}
├── evaluation_final_verified/{evaluation_provenance.json,episode_metrics.jsonl,...}
├── analysis_verified/{pilot_analysis.json,paired_comparisons.csv,pilot_comparison.png}
└── motion_diagnostics/{motion_diagnostics.json,motion_trace.json,motion_trajectories.png}
```

`analysis_verified/pilot_analysis.json` 记录 10 个核心源文件的 SHA-256，包含评估
执行时生成的来源清单。评估器从同一份检查点字节加载模型并计算指纹，执行完毕
后记录原始结果指纹；分析入口核对模型、结果和推理模式的关联。这个机制用于
检测误配和意外改动，不是防恶意篡改的数字签名。

第一次评估没有执行时的来源清单，旧 `analysis/` 仅保留为历史分析，不再作为
新版分析入口的合格输入；没有给旧评估事后补标。新的 `*_verified` 目录与
`analysis_verified/` 是本轮正式交付的可追溯复核产物。
所有输出继续由 `.gitignore` 排除；本次不把模型和个人本地路径上传到仓库。
源码与本文可以提交，科研原始产物保留在本地，没有被删除或覆盖。

新增 `scripts/analyze_training_pilot.py` 只读分析原始结果，生成配对 CSV、JSON
和对比 PNG；拒绝缺回合、重复回合、错种子、场景错配、基线变化、非有限值及
不完整训练预算。参数变化、检查点及完整性核验处有中文说明。
`tests/test_training_pilot.py` 使用临时合成数据测试契约，不把测试数据写入实验目录。
由于原逐回合指标没有保存提前失败的终止原因，本入口只接受失败至步数上限的
试训回合；合法的“全部离线”等提前失败也会被保守拒绝，需要另行诊断，而不是
静默当成完整回合。原通用评估入口仍支持这些场景，没有修改其终止机制。

## 文件改动与验证

本轮只修改/新增 5 个项目文件：

- `scripts/analyze_training_pilot.py`：单种子配对分析、来源及完整性检查、CSV/JSON/PNG。
- `scripts/evaluate_policies.py`：由实际加载的检查点字节生成评估来源清单，既有接口兼容。
- `tests/test_training_pilot.py`：17 项分析、异常输入、来源绑定和真实 CLI 测试。
- `README.md`：补充试训结论、分析命令及适用限制。
- 本文：固定协议、真实结果、诊断、产物位置和完整复现命令。

新增/修改的重要函数、来源绑定、配对统计和保守终止校验均有中文说明；
模型、奖励、损失函数、训练循环和原评价指标没有改动。
测试先复现缺失功能及审查发现的两项缺口，再实现并验证。

最终执行 `python -B -X utf8 -m unittest discover -s tests -q`，**155 项测试全部通过**，
实际耗时 90.726 秒（原有 138 项、新增 17 项）。独立只读审查也复跑了新增
17 项及原学习模块 12 项，合计 29 项通过。
已实际查看训练曲线、配对对比图及单回合运动轨迹图；未发现裁切或图例遮挡。
`git diff --check` 通过，模型和原始结果继续被 `.gitignore` 排除；拟提交文件的
常见密钥、个人绝对路径及邮箱模式检查未发现新增匹配。

## 复现命令

在项目根目录和已安装依赖的 Python 环境中运行，务必使用全新的输出目录。

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
if (Test-Path outputs/pilot_reproduction) { throw '请更换全新输出目录，避免覆盖已有实验' }

# 初始模型和训练入口使用相同架构与种子，避免拿另一份随机模型作为训练前对照。
python -c "from pathlib import Path; import torch; from algorithms import SharedGraphActorCritic; torch.manual_seed(0); m=SharedGraphActorCritic(6,5,2,hidden_dim=64,aggregation='attention'); p=Path('outputs/pilot_reproduction/initial/mappo_checkpoint.pt'); p.parent.mkdir(parents=True,exist_ok=False); torch.save({'model':m.state_dict(),'model_config':m.architecture_config(),'environment_steps':0,'initialization_seed':0},p)"

python scripts/train_mappo.py --updates 100 --rollout-steps 256 --update-epochs 4 --agents 4 --max-steps 300 --hidden-dim 64 --seed 0 --aggregation attention --output-dir outputs/pilot_reproduction/training

$initialCheckpoint = 'outputs/pilot_reproduction/initial/mappo_checkpoint.pt'
$finalCheckpoint = 'outputs/pilot_reproduction/training/mappo_checkpoint.pt'
python scripts/evaluate_policies.py --episodes 20 --agent-counts 4 --max-steps 300 --seed 10000 --checkpoint $initialCheckpoint --packet-loss-rate 0.3 --base-delay 0.2 --delay-jitter 0.1 --noise-std 0.01 --output-dir outputs/pilot_reproduction/evaluation_initial
python scripts/evaluate_policies.py --episodes 20 --agent-counts 4 --max-steps 300 --seed 10000 --checkpoint $finalCheckpoint --packet-loss-rate 0.3 --base-delay 0.2 --delay-jitter 0.1 --noise-std 0.01 --output-dir outputs/pilot_reproduction/evaluation_final

python scripts/analyze_training_pilot.py --input-dir outputs/pilot_reproduction --output-dir outputs/pilot_reproduction/analysis
python -m unittest tests.test_training_pilot -v
python -m unittest discover -s tests -v
```

复核本轮已有结果时，只运行分析入口，并更换新的分析输出目录，例如
`outputs/training_pilot_20261024/analysis_review`，不要重新覆盖训练或评估目录：

```powershell
python scripts/analyze_training_pilot.py --input-dir outputs/training_pilot_20261024 --initial-evaluation outputs/training_pilot_20261024/evaluation_initial_verified --final-evaluation outputs/training_pilot_20261024/evaluation_final_verified --output-dir outputs/training_pilot_20261024/analysis_review
```

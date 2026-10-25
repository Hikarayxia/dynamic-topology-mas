# 本机覆盖记忆实现与试训记录（2026-10-25）

本阶段完成了可选本机覆盖记忆、训练/评估接入、旧检查点兼容和配置错配保护。
功能已能运行，但这次固定预算试训没有获得任务性能收益，保持默认关闭。
本轮没有修改团队奖励、增加角色分工或改变协作补位算法。

## 输入表示与生命周期

`LocalCoverageMemory` 为每个节点维护独立布尔网格，格长固定为 4 米。每次在线
感知只将本机感知圆内的网格中心置为已覆盖；地图末端不足一格时取实际格子中心。
这是一种离散近似，不代表精确覆盖面积，也不表示该区域存在或不存在目标。

模型每次只看到当前所在格及周围 8 格的覆盖标志，再附加 9 个地图有效掩码，
按 `dy=-1..1`、`dx=-1..1` 排列。掩码区分地图外部与地图内尚未覆盖区域。
该摘要固定为 18 维，与节点数量、邻居数量无关。原任务输入中的发现率与剩余时间
保持不变，因此总任务维度由 2 扩为 20；新增的 18 维不读取目标坐标、全局覆盖
或队友轨迹。原先已有的团队发现率仍保留，不能把整个观测称为完全本地信息。

- 重置时先创建每个节点自己的空记忆，再记录初始在线位置的感知范围。
- 每一步移动和节点状态更新后，只记录在线节点的本机感知范围。
- 临时离线时冻结；恢复在线后沿用此前历史。断链不会删除本机记忆。
- 实际移除节点时清除其记忆；重新加入的节点获得新的空记忆。
- 读取观测没有更新副作用，返回的数组不会反向修改内部记忆。

这里验证的是记忆组件对节点增删的处理，不代表固定成员表的补位协议已经
支持任意新成员即插即用。当前也没有覆盖地图通信、联合覆盖分配或循环神经网络。

## 兼容性与初始化

默认 `coverage_memory=False`，原模型输入、配置字典和动作流程保持兼容。
开启时检查点记录 `observation_profile="local_coverage_v1"` 和 `task_dim=20`。
训练环境按模型配置建立观测；评估默认从检查点推断，显式开关相反时在创建
输出目录前报错。手动恢复模型也校验观测版本，防止维度相同而输入含义不同。

训练模型先按旧布局初始化，再扩展任务编码器第一层。共同权重保持相同，
新增输入列初始化为零，扩展过程保留后续 PyTorch 采样随机流。单元测试验证
初始动作在容差内一致，且新增列能够获得非零梯度。

本次隐藏维度为 64，模型参数由 30,342 增至 31,494，增加 1,152 个参数。
两个初始检查点的共同权重逐项一致；新输入权重初始为零，训练后 L2 范数约
0.750108，说明该分支实际参与了优化。这不证明模型学会了有效利用历史。

实验计划、实际训练环境、模型输入和评估场景必须一致。复核发现旧校验可能
接受“训练开、评估关”或反向错配，已先复现再修复并加入双向回归测试。
历史文件缺失记忆字段按关闭解释，读取时规范化，不回写旧文件。

## 实际实验协议

| 项目 | 本轮设置 |
|---|---|
| 算法 | 共享 Graph-MAPPO，attention，Huber 价值损失 |
| 训练种子 | 0，仅一个独立训练种子 |
| 预算 | 100 次更新 × 256 步 = 25,600 环境步；每次更新 4 轮 |
| 环境 | 4 个智能体，50×50 米，6 个目标，dt=0.2，每回合最多 300 步 |
| 训练故障 | 沿用固定时段可恢复链路故障、节点故障；训练通信扰动为 0 |
| 观测 | communicated，邻居状态有效期 1 秒；本轮开启本机覆盖记忆 |
| 测试种子 | 40000–40019，每个场景 20 个配对回合，未用于训练 |
| 测试场景 | 无故障；链路/节点故障；独立通信扰动 |
| 通信扰动场景 | 丢包 0.3、延迟 0.2、延迟抖动 0.1、噪声标准差 0.01 |
| 推理 | 确定性动作，统一评估最后检查点，不按测试奖励挑选中间模型 |

训练真实完成 85 个回合，成功 0 个，最后还有一个未结束回合。前 10 次更新
平均每步奖励为 -0.063776，最后 10 次为 -0.269180；训练日志没有呈现稳定改善。
这里的每步奖励与下面的回合累计奖励口径不同。

无记忆对照复用上一阶段同种子、同预算的 Huber 最终检查点，并在本轮的新种子上
重新评估；没有使用上一阶段种子 20000–20019 的结果冒充本轮配对数据。
核对实际训练配置后，除输出路径、记忆开关和相应输入布局之外均一致。

三个评估目录分别运行初始化模型、无记忆模型和有记忆模型，每次同时运行规则
与补位基线，共实际执行 540 个回合；规则基线逐回合相同，去重后为 300 条记录。

## 真实结果

下表各值均是同场景 20 个回合的均值，“无→有”指训练后的无记忆与有记忆模型。

| 场景 | 累计奖励，无→有 | 碰撞计数，无→有 | 目标发现率，无→有 | 两模型成功率 |
|---|---:|---:|---:|---:|
| 无故障 | -7.935 → -27.318 | 13.50 → 35.85 | 33.33% → 34.17% | 均为 0/20 |
| 链路/节点故障 | -11.364 → -12.525 | 16.15 → 20.35 | 33.33% → 33.33% | 均为 0/20 |
| 通信扰动 | -7.921 → -27.571 | 13.50 → 36.10 | 33.33% → 34.17% | 均为 0/20 |

初始未训练模型在三个场景各成功 1/20，平均累计奖励分别为 2.863、1.874、2.847；
训练后没有优于这个初始化对照。初始化偶然成功也不能视为已训练协作能力。
相同测试种子下，规则策略成功率为 55%、40%、55%，补位基线均为 30%。

碰撞指标是每步接近的智能体对的累计计数，持续接近会重复计数，并非独立物理
事故数。现有故障恢复指标衡量通信健康恢复，不能替代任务能力恢复；聚集在一起
可能较快恢复连通，但同时造成碰撞和低搜索收益。

本次结果只支持“这一实现和这一预算下，单个训练种子未显示收益”。额外输入
同时引入覆盖历史、边界掩码和模型容量变化，不能将差异唯一归因于覆盖历史。
没有跨训练种子方差、显著性检验、已收敛结论或跨规模泛化优势结论。
新种子上无记忆 Huber 也出现更多碰撞，上一阶段的局部改善不能推广到所有场景。

## 文件改动与中文注释

| 文件 | 完成内容 |
|---|---|
| `environments/coverage_memory.py` | 新增本机网格记忆、感知更新、边界掩码、输入校验；说明参数、返回值和近似含义 |
| `environments/continuous_2d.py` | 增加开关和维度查询，将记忆接入初始化、步进、增删节点、重置和观测；解释离线与重置语义 |
| `algorithms/shared_actor_critic.py` | 观测版本与维度约束，检查点保存新版本；保留旧配置字典 |
| `scripts/train_mappo.py` | 增加训练开关和兼容初始化，记录实际配置；说明为何保留共同权重和随机流 |
| `training/mappo.py` | 恢复前校验观测版本，阻止同维度不同语义的模型混用 |
| `evaluation/evaluator.py` | 场景开关、环境接入和模型/场景一致性检查 |
| `scripts/evaluate_policies.py` | 按检查点自动选择记忆输入，显式错配提前拒绝 |
| `scripts/run_ablation.py` | 向训练和评估同步传入开关，默认不改变历史计划参数 |
| `evaluation/ablation.py` | 检查参数、环境、模型和评估一致性；规范化历史缺省值 |
| `scripts/analyze_training_pilot.py` | 读取旧场景时兼容缺省值，检查评估与训练记忆设置 |
| `tests/test_coverage_memory.py` | 新增 18 项本机隔离、边界、生命周期、梯度、CLI、检查点和续跑测试 |
| `tests/test_ablation_reporting.py` | 新增 3 项标签错配、历史兼容和训练/评估反向开关回归测试 |
| `README.md`、本文 | 功能说明、命令、真实实验结果和未完成事项 |

合计 14 个项目文件，重要逻辑有中文注释，原变量名与接口风格保持不变。
新增依赖为零；使用现有 NumPy、PyTorch、Matplotlib。既有奖励、损失定义和
训练循环未另起实现，新增表示沿用原来的编码器及 PPO 优化流程。

## 验证与产物

- 完整回归：`python -B -X utf8 -m unittest discover -s tests -v`，
  **189 项全部通过，102.064 秒**（原 168 项，本轮新增 21 项）。
- 覆盖记忆小规模训练、2/3 节点评估、多组运行及原样续跑由真实 CLI 测试执行。
  节点离线、恢复、移除重加和 2/5 节点观测由组件测试覆盖。
- 完整预算试训及三个独立扰动场景执行成功；检查点参数、日志指标为有限数值。
- 分析核对检查点与评估执行时的来源指纹；对照产物共核验 23 项不同来源文件，
  包括上一阶段的 10 项核心来源文件，均未改变。
- 已检查训练曲线和五组策略对照图，成功率为零的结果正常保留，图例和坐标可读。
- 14 个本轮文件完成语法/文本检查，`git diff --check` 通过；常见私钥、GitHub
  令牌、个人绝对路径和邮箱模式未匹配，训练产物确认为 Git 忽略项。

```text
outputs/coverage_memory_pilot_20261025/
├── initial/mappo_checkpoint.pt
├── training/{run_config.json,training_metrics.jsonl,mappo_checkpoint.pt,training_curves.png}
├── evaluation_initial/{evaluation_provenance.json,episode_metrics.jsonl,...}
├── evaluation_reference_no_memory/{evaluation_provenance.json,episode_metrics.jsonl,...}
├── evaluation_final/{evaluation_provenance.json,episode_metrics.jsonl,...}
├── analysis/{pilot_analysis.json,paired_comparisons.csv,pilot_comparison.png}
└── comparison/{coverage_memory_comparison.json,scenario_summary.csv,paired_memory_minus_no_memory.csv,coverage_memory_comparison.png}
```

模型、日志和原始结果保存在本地，继续由 `.gitignore` 排除，不随源码上传。
`comparison/` 从原始回合重新汇总，只在明确记忆开关差异后比较同场景、同种子，
校验规则基线完全相同；不是另一次训练，也没有改写任何失败回合。

## 运行与复核命令

在项目目录使用安装了项目依赖的 Python，选择全新的输出目录。

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'

# 固定预算训练，启用记忆但保持原团队奖励。
python scripts/train_mappo.py --coverage-memory --value-loss-type huber --updates 100 --rollout-steps 256 --update-epochs 4 --agents 4 --max-steps 300 --hidden-dim 64 --seed 0 --aggregation attention --output-dir outputs/coverage_reproduction/training

# 评估自动识别新观测版本；三个场景采用相同测试种子。
python scripts/evaluate_policies.py --checkpoint outputs/coverage_reproduction/training/mappo_checkpoint.pt --episodes 20 --agent-counts 4 --max-steps 300 --seed 40000 --packet-loss-rate 0.3 --base-delay 0.2 --delay-jitter 0.1 --noise-std 0.01 --output-dir outputs/coverage_reproduction/evaluation_final

# 小预算端到端验证和安全续跑；这不是正式性能实验。
python scripts/run_ablation.py --coverage-memory --value-loss-type huber --aggregations attention --seeds 0 --updates 1 --rollout-steps 8 --update-epochs 1 --train-agents 2 --agent-counts 2,3 --max-steps 8 --hidden-dim 8 --episodes 1 --output-dir outputs/coverage_smoke
python scripts/run_ablation.py --coverage-memory --value-loss-type huber --aggregations attention --seeds 0 --updates 1 --rollout-steps 8 --update-epochs 1 --train-agents 2 --agent-counts 2,3 --max-steps 8 --hidden-dim 8 --episodes 1 --output-dir outputs/coverage_smoke --resume

# 只读复核本轮已保存的初始化/训练后数据；目标分析目录必须不存在。
python scripts/analyze_training_pilot.py --input-dir outputs/coverage_memory_pilot_20261025 --output-dir outputs/coverage_memory_pilot_20261025/analysis_review
python -m unittest tests.test_coverage_memory tests.test_ablation_reporting -v
python -m unittest discover -s tests -v
```

重新生成完整的前后对照时，在训练前以种子 0 调用
`scripts.train_mappo.build_model(64, "attention", coverage_memory=True)`，保存其
`model.state_dict()`、`model.architecture_config()`、`environment_steps=0` 和
`initialization_seed=0` 到 `initial/mappo_checkpoint.pt`，再对初始检查点使用同一
评估命令和种子。初始化文件必须在训练前保存，不能把训练后权重改名为初始模型。

## 尚未解决

学习策略仍未掌握完整搜索与分工，本轮没有完成新的正式多训练种子或跨规模
性能实验。本机记忆是可测试的表示模块，协作补位仍是已有规则基线。
后续应先诊断覆盖输入是否具有有效区分度、共享编码器的梯度影响及碰撞行为，
明确下一项单独对照后再实施，不凭一次失败试训同时更换奖励、角色和网络结构。

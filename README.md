# 动态拓扑多智能体协同仿真

本项目研究动态拓扑下多智能体协同策略的弹性泛化。目前已完成动态拓扑、通信
扰动、连续二维协同搜索环境、规则基线、节点缺失协作补位，以及可训练的图编码
与参数共享 MAPPO 基础实现，并加入基于已投递消息的邻域观测及可学习注意力。
短时训练仅用于验证训练链路，正式结论必须通过
后续对照实验获得。

## 当前功能

- 动态增删节点和边、节点失效/恢复、链路断开/重连；
- 邻接表、加权邻接矩阵、邻居和多跳可达性查询；
- 版本化拓扑变化记录和未来事件调度；
- 连续、离散和空动作统一表示；
- 数量可变的邻域观测及拓扑无关均值聚合基线；
- 智能体状态、抽象智能体和多智能体环境生命周期；
- 单播和邻居广播、通信时延、抖动、丢包、数值噪声与消息过期；
- 动态断链对在途消息的真实影响。
- 连续二维运动、边界裁剪、能量消耗、碰撞与连通性统计；
- 距离驱动通信拓扑、定时链路故障和节点故障/恢复；
- 未知目标探测、共享奖励、成功与时间截断条件；
- 不读取未知目标坐标的分带往返搜索规则基线；
- 固定成员搜索带、心跳超时/二次确认/稳定恢复的节点状态机；
- 基于距离、负载、连通度和能量的去中心化竞价与 gossip 一致选择；
- 节点缺失后的搜索带接管、稳定驻留、恢复交还和补位者切换统计；
- 对邻居顺序与数量不敏感的加权图消息聚合；
- 默认以已投递状态包生成邻域观测，支持消息年龄、缓存过期、乱序保护和显式真值对照；
- 距离权重与消息年龄参与的可学习注意力，以及均值、无邻居输入两种消融模式；
- 参数共享连续 Actor 和在线智能体全局池化的集中式 Critic；
- GAE、PPO 裁剪目标、熵正则、梯度裁剪及动态在线智能体轨迹；
- JSONL 训练指标和 PyTorch 检查点保存；
- 补位响应时间、覆盖恢复时间、未覆盖搜索带比例、切换次数和补位成功率。

## 环境要求

- Python 3.10 或更高版本；
- NumPy 1.24 或更高版本；
- PyTorch 2.2 或更高版本。
- Matplotlib 3.8 或更高版本。

安装为可编辑包：

```powershell
python -m pip install -e .
```

若 Windows 因安装路径过长而安装 PyTorch 失败，建议在较短目录创建虚拟环境：

```powershell
python -m venv D:\venvs\dynamic-topology-mas
D:\venvs\dynamic-topology-mas\Scripts\python.exe -m pip install -e .
```

## 运行测试

```powershell
python -m unittest discover -s tests -v
```

## 运行动态拓扑演示

```powershell
python scripts/demo_dynamic_topology.py
```

## 运行连续二维协同搜索

```powershell
python scripts/run_simulation.py
```

该快速示例使用 4 个规则智能体、6 个随机目标，并在回合中注入短时链路和节点
故障。输出来自实际仿真，不包含预生成实验结果。

## 运行协作补位快速演示

```powershell
python scripts/run_replacement_demo.py --max-steps 40 --seed 7
```

该演示使用 4 个补位智能体、一个临时节点故障和 10% 配置丢包率，打印实际执行
步数、确认缺失任务数、产生响应/恢复覆盖的任务数和实测丢包率。输出完全来自
本次仿真；脚本不会把预期成功写成结果。

## 运行 Graph-MAPPO 训练

先用小规模参数验证完整训练链路：

```powershell
python scripts/train_mappo.py --updates 1 --rollout-steps 8 --update-epochs 1 --agents 3 --hidden-dim 16 --output-dir outputs/quick_validation
```

训练入口默认使用 `attention` 聚合，默认注入一次可恢复链路故障和节点故障，
故障时间随回合长度缩放。通信扰动参数必须显式配置，例如：

```powershell
python scripts/train_mappo.py --updates 100 --rollout-steps 256 --update-epochs 4 --agents 5 --aggregation attention --packet-loss-rate 0.3 --base-delay 0.2 --delay-jitter 0.1 --noise-std 0.01
```

训练指标保存到 `training_metrics.jsonl`，模型与优化器状态保存到
`mappo_checkpoint.pt`，训练曲线保存到 `training_curves.png`。检查点同时保存
网络维度，评估时会优先使用该配置。损失下降本身不等同于任务成功，必须结合
独立验证集的成功率、故障恢复和规模泛化指标评价。

`run_config.json` 保存模型、环境、通信、优化器配置及命令参数。`--seed` 同时
控制模型初始化与环境重置种子；`--no-node-fault`、`--no-link-fault` 可关闭对应故障。
模型构造函数仍默认 `mean`，兼容旧检查点；训练脚本默认值为 `attention`。

## 通信观测与注意力消融

- `--observation-mode communicated`（默认）：每步广播本机位置、速度、能量的
  数值快照，通过通信总线后才进入接收方缓存；未收到状态包就没有对应邻居特征。
- `--neighbor-state-ttl 1.0`：缓存最多保留 1 秒；失联后保留未过期快照，
  不读取真实拓扑来提前判定邻居失效。延迟超过 TTL 的包不进入观测，旧包不能覆盖新包。
  直接移除节点也不会清空其他接收方的有效快照，缓存仍从消息生成时刻计时；
  该接口不模拟成功送达的退出通知。`oracle` 对照模式仍立即反映节点移除。
- `--observation-mode oracle`：显式恢复旧版实时真值邻居输入，仅用于理想信息
  对照和旧实验复现。该模式下丢包和时延不会改变邻居特征，不能证明通信鲁棒性。
- `--aggregation mean|attention|none`：距离加权均值、可学习注意力、无邻居输入。
  注意力输入包括自身及邻居编码、由接收快照估计的距离权重和消息年龄；并未
  实现信噪比建模或真实链路丢包率估计，也不保证未经训练的模型会降低旧消息权重。

状态噪声施加在原始 `[x, y, vx, vy, energy]` 数值载荷上，各分量使用相同配置的
标准差；这是一种简化的测量扰动模型。非状态协议消息继续进入各智能体收件箱，
状态包不占用规则补位的 gossip 收件箱。通信统计现在也包括这些状态包。
两类消息使用独立随机流，额外状态广播不会改变同一种子下 gossip 的丢包、
抖动和噪声样本；当前总线未建模带宽竞争或拥塞。

注意：任务输入仍包括全队目标发现比例，规则策略还使用环境提供的成员索引和
在线数量。当前并非完全局部信息的实物部署协议，完全去中心化任务信息仍需后续研究。

批量运行三种聚合器、三个独立训练种子，并在 5、10、20、50 个智能体上评估：

```powershell
python scripts/run_ablation.py --dry-run --output-dir outputs/ablation_v1
python scripts/run_ablation.py --output-dir outputs/ablation_v1
```

第一条只保存实验计划，第二条开始顺序运行 9 组训练；每组默认 100 次更新，
每个测试场景 20 回合，可能耗时较长。各组输出独立保存，已有实验不会被覆盖。
训练种子默认 0、1、2，测试种子从 10000 开始，所有策略使用配对测试种子。
`none` 是本框架的无邻居消融，不代表完整复现某个外部标准 MAPPO 实现。

### 多种子结果汇总

完整实验矩阵运行结束后，`run_ablation.py` 自动在实验目录的 `summary/` 下保存：

- `ablation_summary.csv`：按聚合器、策略、场景和规模分组的均值、样本标准差及有效样本数；
- `ablation_summary.json`：相同统计量，以及来源目录、训练配置、测试配置和统计提示；
- `ablation_comparison.png`：成功率、累计奖励、最终目标发现比例、故障恢复时间对比图。

也可以对已有完整实验单独汇总，无需重新训练：

```powershell
python scripts/summarize_ablation.py --input-dir outputs/ablation_v1
```

如已有汇总，使用新目录保存复核结果，避免覆盖先前记录：

```powershell
python scripts/summarize_ablation.py --input-dir outputs/ablation_v1 --output-dir outputs/ablation_v1/summary_review
```

统计时先在每个训练种子内部汇总评估回合，再对独立训练种子的结果等权计算
均值和样本标准差（分母为有效种子数减一），不把多个评估回合当作多个训练种子。
仅一个有效种子时标准差为空，图中不画误差棒；未发生恢复等事件的指标保持为空，
CSV 的 `_count` 和 JSON 的 `count` 记录该指标的有效样本数。恢复时间等条件均值
应结合恢复率一起解读，不能将未恢复回合计成零秒。

规则基线不依赖模型训练种子。同一配对测试任务下的重复基线先核对结果一致，
再去重为一份；其 `training_seed_count` 为 0，`replicate_count` 为 1，不估计
跨训练种子的标准差。缺失训练组、重复回合、测试种子错配、实际训练预算不足、
不一致配置或非有限指标都会导致汇总失败，不生成不完整的比较结果。
输入必须包含原实验计划、每组训练配置与训练日志、评估场景配置及逐回合指标；
只生成计划的 `--dry-run` 目录不能当作已经完成的实验汇总。

### 中断后的安全续跑

默认仍拒绝覆盖已有实验。对于上面使用默认参数启动的实验，可先预览，再续跑：

```powershell
python scripts/run_ablation.py --output-dir outputs/ablation_v1 --resume --dry-run
python scripts/run_ablation.py --output-dir outputs/ablation_v1 --resume
```

首次运行如显式修改了训练预算、种子、规模或通信参数，续跑时必须沿用原参数。
程序会读取原 `ablation_plan.json` 核对配置，不接受把不同预算混入同一实验矩阵；
完整目录迁移后，旧程序和输出路径可以变化，但所有阶段均使用当前项目生成的
命令，不执行计划文件中保存的旧程序路径。

- 完整训练必须同时通过配置、日志采样预算、检查点结构与有限参数检查，才会跳过。
- 完整评估必须通过场景、配对测试种子及全部回合检查，才会跳过。
- 训练不完整时，将旧训练和对应旧评估移到该训练组的 `attempts/` 后重新运行；
  仅评估不完整时，只归档并重跑评估。归档保留原文件，不删除已有数据。
- 汇总与当前数据一致且 CSV、PNG 完整时复用，否则把旧汇总移入根目录的
  `attempts/summary-*` 后重建。独立汇总入口仍要求使用新输出目录。
- `ablation_events.jsonl` 追加记录阶段开始、完成、跳过、归档及失败事件，时间为 UTC。
- 同一实验目录使用操作系统文件锁，防止两个实验入口同时写入。`.ablation.lock`
  文件会保留，但进程退出会释放锁，无需手工删除。预览不改动已有结果、计划或事件日志。

这里实现的是**训练/评估阶段级续跑**。单次训练若尚未完整保存日志和检查点，
该训练组会从头重跑；不声称恢复到中断瞬间的 PPO 轨迹、环境或随机数状态。

## 运行弹性与规模泛化评估

评估规则基线和协作补位策略：

```powershell
python scripts/evaluate_policies.py --episodes 10 --agent-counts 3,4,6
```

同时加入已保存的学习策略：

```powershell
python scripts/evaluate_policies.py --episodes 10 --agent-counts 3,4,6 --checkpoint outputs/mappo_checkpoint.pt
```

增加独立通信扰动场景（无故障组仍采用理想通信，链路/节点故障组单独评估）：

```powershell
python scripts/evaluate_policies.py --episodes 20 --agent-counts 5,10,20,50 --checkpoint outputs/mappo_checkpoint.pt --packet-loss-rate 0.3 --base-delay 0.2 --delay-jitter 0.1 --noise-std 0.01
```

`scenario_config.json` 保存各场景的完整配置，便于复现。

每种规模都会运行无故障场景和链路/节点故障场景，并保存：

- `episode_metrics.jsonl`：逐回合原始指标；
- `scenario_summary.csv/json`：成功率、收敛步数、累计奖励、一致性误差、
  连通率、碰撞、故障恢复率、恢复时间、相对无故障性能下降和补位专项指标；
- `evaluation_summary.png`：不同策略、规模和故障条件的常规对比图；
- `replacement_summary.png`：补位响应、覆盖恢复、未覆盖比例和切换次数专项图。

为保证公平，无故障和故障场景使用相同种子序列；每个回合重新创建环境与智能
体，避免控制器内部状态泄漏。未成功回合的收敛步数、未恢复故障的恢复时间保持
为空，不会以最大步数或零值代替。

协作补位是可解释、确定性的分布式规则基线，不是全局最优分配器，也不等同于
经过训练的鲁棒控制策略。当前 gossip 只能沿当时可用的通信边传播；网络分区时
各连通分量会独立形成局部认知，重连后再通过新消息收敛。因此正式实验应分别
报告丢包、分区时长和拓扑规模，不应只给出单一成功率。

## 主要接口

- `core.topology.DynamicTopology`：动态图和邻接关系；
- `core.action.Action`：统一动作；
- `core.observation.Observation`：自身、邻域和任务观测；
- `core.message.Message`：通信消息；
- `core.agent.BaseAgent`：智能体抽象；
- `core.environment.BaseEnvironment`：环境抽象；
- `interaction.communication.CommunicationBus`：动态拓扑通信总线。
- `environments.continuous_2d.Continuous2DSearchEnv`：连续协同搜索环境；
- `agents.rule_based_agent.RuleBasedSearchAgent`：规则搜索基线；
- `coordination.replacement.ReplacementCoordinator`：缺失确认、竞价、gossip、
  稳定恢复与任务交还；
- `agents.replacement_agent.ReplacementSearchAgent`：执行搜索带补位的规则智能体；
- `algorithms.graph_encoder.GraphObservationEncoder`：可变邻域图编码器；
- `algorithms.shared_actor_critic.SharedGraphActorCritic`：共享 Actor/集中 Critic；
- `training.mappo.MAPPOTrainer`：MAPPO 采样、GAE 与 PPO 更新器；
- `agents.learning_agent.SharedPolicyAgent`：共享策略分散执行封装；
- `evaluation.evaluator`：独立场景、规则/补位/学习策略与规模泛化评估；
- `evaluation.metrics`：协同、连通、补位与故障恢复指标；
- `evaluation.plotting`：训练及评估 PNG 图表。

总体设计与当前实现状态见 `docs/architecture.md`。`legacy/` 仅保存不可运行的
早期原型，不属于正式代码。

2026 年 10 月 23 日的仓库核查、开发验证和下一阶段检查点见
[`docs/progress_2026-10-23.md`](docs/progress_2026-10-23.md)。

2026 年 10 月 24 日的多种子汇总、安全续跑及真实验证记录见
[`docs/progress_2026-10-24.md`](docs/progress_2026-10-24.md)。

### 单种子试训与独立配对评估

25,600 步的单种子试训已完成；独立测试没有得到成功协作策略，并出现贴边聚集
和碰撞增加。完整协议、真实结果、限制及复现命令见
[`docs/training_pilot_2026-10-24.md`](docs/training_pilot_2026-10-24.md)。
这不是已收敛的模型，也不作为跨规模泛化或聚合器优势的证据。

已保存初始模型、训练产物及前后独立评估后，可以只读生成配对分析：

```powershell
python scripts/analyze_training_pilot.py --input-dir outputs/training_pilot_20261024 --initial-evaluation outputs/training_pilot_20261024/evaluation_initial_verified --final-evaluation outputs/training_pilot_20261024/evaluation_final_verified --output-dir outputs/training_pilot_20261024/analysis_review
```

输入结构为 `initial/`、`training/`、`evaluation_initial/`、`evaluation_final/`。
分析前会核对训练预算、检查点、测试场景、种子独立性和回合完整性；规则基线
逐回合核对一致后去重。输出包含 JSON、配对差值 CSV 和对比图，目标目录必须
不存在，不覆盖原始实验。只有一个训练种子，不估计跨训练种子方差或显著性。
评估必须由新版入口在执行时生成 `evaluation_provenance.json`，将模型指纹与
原始结果关联；缺失清单的旧结果不会被事后补标。本分析入口保守拒绝无法核验
终止原因的提前失败回合，原通用评估入口的故障终止行为保持不变。

### 可选稳健价值损失

训练入口和多种子入口支持 `--value-loss-type mse|huber`，默认 `mse` 保持原行为。
Huber 使用固定 delta=1，限制大价值误差的梯度；不改变团队奖励，也不是避碰
控制器或安全保证。实际类型记录在配置与检查点中，续训拒绝静默切换损失函数。

```powershell
python scripts/train_mappo.py --updates 100 --rollout-steps 256 --agents 4 --max-steps 300 --aggregation attention --value-loss-type huber --output-dir outputs/huber_trial/training
```

单种子、同预算及新配对种子的真实对照中，Huber 减少了碰撞并提高累计奖励，
但任务成功率仍为 0；不能据此声称算法已收敛或完成协作搜索。协议、结果、
兼容性说明与复现命令见
[`docs/training_stability_2026-10-24.md`](docs/training_stability_2026-10-24.md)。


## 价值梯度隔离对照（2026-10-25）

训练支持 `--critic-encoder-mode shared|detached`，默认 `shared` 保持已有行为。
`detached` 将 Critic 的输入编码在计算价值损失前 detach，保留 Actor 到编码器的
训练路径；不新增网络参数，不改变观测、动作、奖励或推理的前向值。

```powershell
python scripts/train_mappo.py --updates 100 --rollout-steps 256 --update-epochs 4 --agents 4 --max-steps 300 --hidden-dim 64 --seed 0 --aggregation attention --value-loss-type huber --critic-encoder-mode detached --output-dir outputs/critic_detached/training
```

做对照时应显式设置相同的 rollout 步数、PPO epochs 和所有其他参数；批量入口
`scripts/run_ablation.py` 也接收此选项，默认共享模式与旧计划保持续跑兼容。
检查点恢复会拒绝梯度模式错配，旧检查点缺失该字段时解释为 `shared`。

该选项只隔离价值损失到编码器的直接梯度，不等同于两个完全独立的网络：
价值预测仍影响 GAE，Actor 更新仍改变 Critic 所用的表示，全局梯度裁剪也仍会
影响共同优化。它不增加搜索记忆、角色分工或安全控制能力。
本轮配对试训、适用限制与原始产物说明见
[`docs/critic_isolation_2026-10-25.md`](docs/critic_isolation_2026-10-25.md)。

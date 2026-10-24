# 训练退化诊断与稳健价值损失对照：2026-10-24

## 本轮结论

新增了可选 Huber 价值损失，**默认仍为原来的 MSE**，没有替换 GNN、Actor/Critic
结构、团队奖励、故障机制、动作接口或评价指标。

相同初始权重、训练种子和 25,600 步预算下，Huber 模型在新的配对评估中减少了
碰撞、提高了目标发现率和累计奖励。三个场景的任务成功率仍然均为 0。
因此本轮成果是“训练稳健化选项及局部行为改善”，不是已经解决协作搜索，
也不是新增了有安全保证的避碰控制器。

## 为什么选择这个小范围对照

先读取并复核了旧试训，再在内存中诊断 PPO 的概率比与共享编码器梯度：

- 初始模型、环境诊断种子 30000、256 步真实轨迹中，未更新参数时重算动作
  log probability 的最大绝对误差为 2.3842e-6，未观察到明显的旧/新概率错配。
- 同批轨迹的 Actor 共享编码器梯度范数为 0.06317；加权 MSE 价值梯度范数为
  0.81921，约为前者的 12.97 倍。只执行一次 Critic 更新也会使 Actor 的
  确定性动作改变，最大变化约 0.00227，证明共享编码器存在直接影响路径。
- 同一批轨迹改用 Huber 后，该梯度范数为 0.59655，约为策略梯度的 9.44 倍；
  仍非零，也不能由一个批次证明它是全部退化的唯一根因。

Huber(delta=1) 在小误差段采用二次项，大误差段采用线性项；本实现使用
`smooth_l1_loss(beta=1)`，此时与 Huber(delta=1) 等价，预测误差的尾部梯度有界。
数学定义见 [PyTorch 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html)。
测试还以手算损失和梯度验证了本机实际安装版本的行为，没有为此新增依赖。

必须注意，标准 Huber 在二次区间也有 0.5 因子，而原 MSE 为完整平方误差。
本次改变的是整个价值损失函数，不能将效果进一步归因为“只有离群点处理变化”，
也不能直接比较 MSE 与 Huber 的原始 Value loss 数字来宣称价值预测更准确。
共享编码器梯度依然存在，训练曲线中仍有较大波动。

### 没有擅自修改的部分

原 GAE 把回合结束时的后续价值置零。外部采样截断与任务内有限期限的自举语义
不同，不能只看到 `truncated` 字段就把当前实现认定为错误。
现有观测包含剩余时间；本轮保留原期限与 GAE 行为，未改变任务定义。
若后续要改成无限时域任务的外部截断，需要先明确协议。相关区别见
[Gymnasium 官方时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。

本轮也没有改变在线人数的 PPO 样本加权、确定性评估方式、邻居观测、能量模型
或规则补位行为，没有增加覆盖历史、角色分工或隐藏的规则动作替代。

## 固定实验协议

- 使用旧试训的同一份初始模型文件，逐字节核对一致，初始化及训练种子均为 0。
- 4 个智能体、50×50 地图、6 个未知目标、每回合最多 300 步、dt=0.2。
- attention 聚合、隐藏维度 64；100 次更新、每次 256 步、4 个 PPO epoch。
- 训练环境参数、奖励、故障、通信参数及其他优化设置均与旧 MSE 试训相同。
  实际配置去掉输出路径和新增损失类型字段后完全一致。
- Huber 主训练实际用时 149.460 秒，完成 85 个回合，成功数为 0。
  耗时来自启动命令的 Stopwatch 执行输出，不是训练 JSONL 内的字段。
- 新评估种子为 20000–20019，不复用旧的 10000–10019 作为本次新对照的独立证据。
  前述梯度诊断种子 30000 也不在此评估集合中。
- 初始模型、旧 MSE 模型和新 Huber 模型分别评估；每组包含正常、链路/节点故障、
  通信扰动三个场景，以及规则、规则补位、学习三种策略，每个组合 20 回合。
- 通信扰动为丢包率 0.3、基础延迟 0.2、抖动 0.1、噪声标准差 0.01；
  正常和拓扑故障组继续使用理想通信，以保持可归因性。
- 三套评估共执行 540 个回合；重复规则基线逐回合核对一致后去重，保留
  300 个“模型/策略 × 场景 × 种子”记录，不将其当成多个独立训练种子。

## 同一批新测试种子的实际结果

下表为学习模型，数值是每场景 20 个回合的均值：

| 场景 | MSE→Huber 累计奖励 | MSE→Huber 碰撞计数 | MSE→Huber 目标发现率 | 两种模型成功率 |
|---|---:|---:|---:|---:|
| 正常 | -21.893 → 2.021 | 24.00 → 1.65 | 22.50% → 31.67% | 均为 0/20 |
| 链路/节点故障 | -18.749 → 3.644 | 20.50 → 0.25 | 22.50% → 32.50% | 均为 0/20 |
| 通信扰动 | -22.303 → 2.212 | 24.45 → 1.45 | 22.50% → 31.67% | 均为 0/20 |

碰撞是每步接近的智能体对累计计数，持续接近会多次记账，不是独立物理事故数。
Huber 在正常场景的该计数较 MSE 减少约 93.1%，但没有消除碰撞，也未完成任务。
同批测试的规则策略成功率分别为 50%、45%、50%，说明仍有明显任务性能差距。

Huber 训练前 10 次更新的平均每步奖励为 -0.06746，最后 10 次为 -0.06405。
原 MSE 同口径为 -0.07057 和 -0.71699。训练每步奖励与上表回合累计奖励不同，
不能相互直接比较。这里只有一个训练种子，不报告跨训练种子标准差或显著性，
不宣称收敛、普适安全性或跨规模泛化优势。

## 源码改动与中文说明

- `training/mappo.py`：新增 `value_loss_type` 与统一价值损失计算；默认保留 MSE，
  解释 Huber 的用途、参数和返回值；恢复检查点前核对损失类型，旧无字段按 MSE。
- `scripts/train_mappo.py`：新增 `--value-loss-type mse|huber`，实际类型写入配置和检查点。
- `scripts/run_ablation.py`：支持把 Huber 选项传入子训练；默认 MSE 不新增计划参数，
  以兼容没有此参数的旧实验计划续跑。
- `evaluation/ablation.py`：检查命令标签与实际训练配置的损失类型，缺失字段按历史 MSE。
  配置签名也统一补齐默认值，避免旧完成组与新版 MSE 重跑组被误判为不同协议。
- `tests/test_training_stability.py`：12 项手算数学、尾部梯度、真实 PPO 更新、CLI、
  检查点恢复、旧计划兼容及错配拒绝测试；合成数据只写临时目录。
- `tests/test_ablation_reporting.py`：新增 1 项新旧默认 MSE 训练组混合汇总回归测试。
- `README.md` 与本文：补充命令、对照结果与未解决问题。

重要配置、共享编码器影响、损失函数和恢复兼容处均有中文说明。
本轮共修改/新增 8 个项目文件，没有新增安装依赖。

## 本地原始产物

```text
outputs/training_stability_huber_20261024/
├── initial/mappo_checkpoint.pt
├── training/{run_config.json,training_metrics.jsonl,mappo_checkpoint.pt,training_curves.png}
├── evaluation_initial/{evaluation_provenance.json,episode_metrics.jsonl,...}
├── evaluation_reference_mse/{evaluation_provenance.json,episode_metrics.jsonl,...}
├── evaluation_final/{evaluation_provenance.json,episode_metrics.jsonl,...}
├── analysis/{pilot_analysis.json,paired_comparisons.csv,pilot_comparison.png}
├── comparison/{value_loss_comparison.json,scenario_summary.csv,paired_huber_minus_mse.csv,value_loss_comparison.png}
└── diagnostics/optimization_diagnostics.json
```

分析入口沿用上一轮的来源校验，所有评估都有执行时的检查点和原始结果指纹。
`comparison/` 是本轮从真实原始回合生成的配对复核产物；校验三组场景与基线一致、
初始化权重一致、训练协议仅损失类型不同后再作差。旧试训 10 个核心来源文件的
SHA-256 全部未变。模型、日志和原始结果继续由 `.gitignore` 排除，不上传。
旧试训的这 10 项来源取自 `outputs/training_pilot_20261024/analysis_verified/pilot_analysis.json`，
不是未包含来源清单的早期 `analysis/` 历史版本。

## 最终验证

- 最终完整回归：`python -B -X utf8 -m unittest discover -s tests -q`，
  **168 项全部通过**，实际耗时 98.303 秒（原有 155 项，本轮新增 13 项）。
- 数学/CLI/恢复测试先观察缺失功能失败，再实现；审查发现的新旧 MSE 汇总问题
  也先复现，再增加缺省值规范化与回归用例。
- 相关的消融汇总及训练稳健化测试共 29 项通过；最终代码重新分析本轮原始数据，
  得到与已保存 JSON 完全一致的结果。
- 当前分析 10 项、MSE/Huber 对照 9 项、旧试训 10 项来源哈希全部匹配。
- 实际检查点为 25,600 步、配置为 Huber，模型参数均为有限数值。
- 已查看训练曲线和 MSE/Huber 对照图；图例、坐标和成功率为零的结果正常显示。
- `git diff --check` 通过；拟提交新增文件的常见密钥/个人绝对路径/邮箱模式未发现匹配。

## 运行与复核

在项目目录使用已安装依赖的 Python，选择全新的输出目录。

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'

# 单独训练稳健化对照；其他参数与 MSE 试训一致。
python scripts/train_mappo.py --updates 100 --rollout-steps 256 --update-epochs 4 --agents 4 --max-steps 300 --hidden-dim 64 --seed 0 --aggregation attention --value-loss-type huber --output-dir outputs/huber_reproduction/training

# 规则及最终学习策略的独立评估；不要覆盖已有评估目录。
python scripts/evaluate_policies.py --episodes 20 --agent-counts 4 --max-steps 300 --seed 20000 --checkpoint outputs/huber_reproduction/training/mappo_checkpoint.pt --packet-loss-rate 0.3 --base-delay 0.2 --delay-jitter 0.1 --noise-std 0.01 --output-dir outputs/huber_reproduction/evaluation_final

# 复核本轮已有的完整前后评估，分析目标目录必须尚不存在。
python scripts/analyze_training_pilot.py --input-dir outputs/training_stability_huber_20261024 --output-dir outputs/training_stability_huber_20261024/analysis_review

python -m unittest tests.test_training_stability -v
python -m unittest discover -s tests -v
```

若要重新完整复现初始化和前后评估，使用上一轮
[`training_pilot_2026-10-24.md`](training_pilot_2026-10-24.md) 的流程，训练增加
`--value-loss-type huber`，评估统一使用新种子 20000，所有输出使用新目录。
多种子入口也支持 `--value-loss-type huber`；这只是提供配置能力，本轮没有
执行或声称完成新的正式多种子实验。

## 仍需完成

局部碰撞退化已有改善，但学习策略仍未学到完成搜索所需的覆盖与分工。
下一阶段需要明确覆盖历史/角色表示、共享表示的梯度隔离或安全控制层的方案，
再逐项对照，不把规则扫描伪装成学习成果。

"""多种子汇总的统计、完整性检查与命令行测试；合成数据仅用于单元测试。"""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from environments import Continuous2DConfig
from evaluation.evaluator import ExperimentScenario
from interaction import CommunicationConfig
from training import MAPPOConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AblationReportingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    @staticmethod
    def _json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _fixture(self, aggregations=("mean",), seeds=(0, 1, 2)):
        """构造完整协议输入；奖励与成功数使用可手算的固定测试值。"""

        plan = []
        for aggregation in aggregations:
            for seed in seeds:
                run = self.root / aggregation / f"seed_{seed}"
                arguments = dict(
                    updates=1, rollout_steps=4, update_epochs=1, agents=3,
                    max_steps=4, hidden_dim=8, seed=seed, aggregation=aggregation,
                    observation_mode="communicated", neighbor_state_ttl=1.0,
                    packet_loss_rate=0.0, base_delay=0.0, delay_jitter=0.0,
                    noise_std=0.0, link_fault=True, node_fault=True,
                    output_dir=str(run / "training"),
                )
                train = [sys.executable, "train_mappo.py"]
                for key in ("updates", "rollout_steps", "update_epochs", "agents",
                            "max_steps", "hidden_dim", "seed", "aggregation"):
                    train.extend(["--" + key.replace("_", "-"), str(arguments[key])])
                evaluate = [
                    sys.executable, "evaluate_policies.py", "--seed", "10000",
                    "--episodes", "2", "--agent-counts", "3", "--max-steps", "4",
                ]
                plan.append(dict(aggregation=aggregation, seed=seed, train=train, evaluate=evaluate))
                self._json(run / "training/run_config.json", dict(
                    arguments=arguments,
                    model=dict(self_dim=6, neighbor_dim=5, task_dim=2, action_dim=2,
                               hidden_dim=8, aggregation=aggregation),
                    environment=asdict(Continuous2DConfig(max_steps=4, n_targets=6)),
                    communication=asdict(CommunicationConfig()),
                    training=asdict(MAPPOConfig(rollout_steps=4, update_epochs=1)),
                ))
                self._jsonl(run / "training/training_metrics.jsonl", [dict(
                    update=1, environment_steps=4, mean_team_reward=0.0,
                    actor_loss=0.0, value_loss=1.0, entropy=1.0, total_loss=0.49,
                    episodes_finished=1, successful_episodes=0,
                )])
                self._json(run / "evaluation/scenario_config.json", [asdict(
                    ExperimentScenario("nominal", 3, episodes=2, base_seed=10000, max_steps=4)
                )])
                records = []
                for policy in ("rule", "replacement", "learned"):
                    for episode in range(2):
                        success = policy == "learned" and episode < seed
                        records.append(dict(
                            scenario="nominal", policy=policy, episode=episode,
                            seed=10000 + episode, agent_count=3, success=success,
                            steps=4, convergence_steps=4 if success else None,
                            cumulative_team_reward=(4 * seed + 2 * episode)
                            if policy == "learned" else 2 + episode,
                            mean_connectivity=1.0, mean_consistency_error=0.1,
                            collisions=0, final_detection_ratio=1.0 if success else 0.5,
                            fault_recovery_time=None, topology_performance_drop=None,
                            replacement_response_time=None, coverage_recovery_time=None,
                            mean_uncovered_lane_ratio=0.0, replacement_switches=0,
                            replacement_success_rate=None,
                        ))
                self._jsonl(run / "evaluation/episode_metrics.jsonl", records)
        self._json(self.root / "ablation_plan.json", plan)

    def _report(self):
        from evaluation.ablation import summarize_ablation
        return summarize_ablation(self.root)

    def _edit_json(self, relative, edit):
        path = self.root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        edit(value)
        self._json(path, value)

    def _edit_episodes(self, edit):
        path = self.root / "mean/seed_1/evaluation/episode_metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        edit(rows)
        self._jsonl(path, rows)

    def test_cli_creates_report_from_existing_experiment_matrix(self):
        self._fixture()
        result = subprocess.run([
            sys.executable, "-B", "-X", "utf8",
            str(PROJECT_ROOT / "scripts/summarize_ablation.py"),
            "--input-dir", str(self.root),
        ], capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root / "summary/ablation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["completed_runs"], 3)

    def test_statistics_use_training_seed_means_not_pooled_episodes(self):
        self._fixture()
        report = self._report()
        learned = next(row for row in report["rows"] if row["policy"] == "learned")
        reward = learned["metrics"]["mean_cumulative_reward"]
        # 三个种子分别得到 1、5、9；样本标准差为 4，而不是六回合的混合标准差。
        self.assertEqual(reward, dict(mean=5.0, std=4.0, count=3))
        self.assertEqual(learned["metrics"]["success_rate"], dict(mean=0.5, std=0.5, count=3))
        self.assertEqual(learned["training_seed_count"], 3)

    def test_old_and_new_default_mse_runs_can_be_summarized_together(self):
        self._fixture(seeds=(0, 1))
        self._edit_json("mean/seed_0/training/run_config.json",
                        lambda value: value["training"].pop("value_loss_type", None))
        self._edit_json("mean/seed_1/training/run_config.json",
                        lambda value: value["arguments"].update(value_loss_type="mse"))
        # 中断矩阵仅重跑一组时，缺省的旧 MSE 与显式的新 MSE 是同一协议。
        try:
            report = self._report()
        except ValueError as error:
            self.fail(f"同一 MSE 协议不应因新旧字段差异被拒绝: {error}")
        self.assertEqual(report["completed_runs"], 2)
        learned = next(row for row in report["rows"] if row["policy"] == "learned")
        self.assertEqual(learned["training_seed_count"], 2)

    def test_missing_events_and_single_seed_do_not_become_zero_variance(self):
        self._fixture(seeds=(0,))
        report = self._report()
        learned = next(row for row in report["rows"] if row["policy"] == "learned")
        self.assertIsNone(learned["metrics"]["success_rate"]["std"])
        self.assertEqual(learned["metrics"]["mean_recovery_time"], dict(mean=None, std=None, count=0))
        self.assertTrue(report["warnings"])

    def test_identical_baselines_are_not_counted_as_independent_training_runs(self):
        self._fixture(aggregations=("mean", "attention"))
        report = self._report()
        baseline = [row for row in report["rows"] if row["policy"] == "rule"]
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["replicate_count"], 1)
        self.assertEqual(baseline[0]["training_seed_count"], 0)
        self.assertIsNone(baseline[0]["metrics"]["mean_cumulative_reward"]["std"])

    def test_inconsistent_baseline_results_are_rejected(self):
        self._fixture()
        self._edit_episodes(lambda rows: rows[0].update(cumulative_team_reward=999))
        with self.assertRaisesRegex(ValueError, "基线"):
            self._report()

    def test_missing_planned_run_is_rejected(self):
        self._fixture(aggregations=("mean", "attention"))
        self._edit_json("ablation_plan.json", lambda plan: plan.pop())
        with self.assertRaisesRegex(ValueError, "缺少|不完整"):
            self._report()

    def test_duplicate_training_seed_is_rejected(self):
        self._fixture()
        self._edit_json("ablation_plan.json", lambda plan: plan.append(plan[0]))
        with self.assertRaisesRegex(ValueError, "重复"):
            self._report()

    def test_missing_output_file_is_rejected(self):
        self._fixture()
        (self.root / "mean/seed_1/evaluation/episode_metrics.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "episode_metrics"):
            self._report()

    def test_training_configuration_mismatch_is_rejected(self):
        self._fixture()
        self._edit_json("mean/seed_1/training/run_config.json",
                        lambda config: config["communication"].update(packet_loss_rate=0.5))
        with self.assertRaisesRegex(ValueError, "配置"):
            self._report()

    def test_incomplete_training_budget_is_rejected(self):
        self._fixture()
        path = self.root / "mean/seed_1/training/training_metrics.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["environment_steps"] = 2
        self._jsonl(path, [row])
        with self.assertRaisesRegex(ValueError, "训练|预算"):
            self._report()

    def test_mismatched_evaluation_seeds_are_rejected(self):
        self._fixture()
        self._edit_json("mean/seed_1/evaluation/scenario_config.json",
                        lambda rows: rows[0].update(base_seed=20000))
        with self.assertRaisesRegex(ValueError, "种子|配置"):
            self._report()

    def test_duplicate_and_missing_episodes_are_rejected(self):
        self._fixture()
        self._edit_episodes(lambda rows: rows.append(rows[-1]))
        with self.assertRaisesRegex(ValueError, "重复"):
            self._report()
        self._fixture()
        self._edit_episodes(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "缺失|数量|不完整"):
            self._report()

    def test_nonfinite_metric_is_rejected(self):
        self._fixture()
        self._edit_episodes(lambda rows: rows[-1].update(cumulative_team_reward=float("nan")))
        with self.assertRaisesRegex(ValueError, "有限|NaN"):
            self._report()

    def test_required_reward_cannot_be_replaced_by_null(self):
        self._fixture()
        self._edit_episodes(lambda rows: rows[-1].update(cumulative_team_reward=None))
        with self.assertRaisesRegex(ValueError, "cumulative_team_reward"):
            self._report()

    def test_overflowing_json_number_is_rejected_with_source_path(self):
        self._fixture()
        path = self.root / "mean/seed_1/training/run_config.json"
        text = path.read_text(encoding="utf-8").replace('"packet_loss_rate": 0.0',
                                                      '"packet_loss_rate": 1e999')
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "run_config.json.*有限"):
            self._report()

    def test_outputs_preserve_sources_and_refuse_overwrite(self):
        from evaluation.ablation import save_ablation_report

        self._fixture()
        source = self.root / "mean/seed_1/evaluation/episode_metrics.jsonl"
        before = source.read_bytes()
        report = self._report()
        paths = save_ablation_report(report, self.root / "summary")
        self.assertEqual(source.read_bytes(), before)
        with paths["csv"].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        learned = next(row for row in rows if row["policy"] == "learned")
        self.assertEqual(float(learned["mean_cumulative_reward_std"]), 4.0)
        self.assertTrue(paths["figure"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        saved = paths["json"].read_bytes()
        with self.assertRaises(FileExistsError):
            save_ablation_report(report, self.root / "summary")
        self.assertEqual(paths["json"].read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()

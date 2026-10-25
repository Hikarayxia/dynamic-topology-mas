"""试训分析的数据契约测试；手算样例仅写入临时目录，不充当实验结果。"""

from dataclasses import asdict
import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from algorithms import SharedGraphActorCritic
from environments import Continuous2DConfig
from evaluation.evaluator import ExperimentScenario
from evaluation.metrics import EpisodeMetrics
from interaction import CommunicationConfig
from training import MAPPOConfig, TrainingMetrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/analyze_training_pilot.py"


class TrainingPilotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        arguments = dict(seed=0, updates=2, rollout_steps=2, update_epochs=1,
                         agents=3, max_steps=2, hidden_dim=8, aggregation="attention",
                         observation_mode="communicated", neighbor_state_ttl=1.0)
        self._json(self.root / "training/run_config.json", dict(
            arguments=arguments,
            model=dict(self_dim=6, neighbor_dim=5, task_dim=2, action_dim=2,
                       hidden_dim=8, aggregation="attention"),
            training=asdict(MAPPOConfig(rollout_steps=2, update_epochs=1)),
            environment=asdict(Continuous2DConfig(max_steps=2, n_targets=6)),
            communication=asdict(CommunicationConfig()),
        ))
        self._jsonl(self.root / "training/training_metrics.jsonl", [
            asdict(TrainingMetrics(1, 2, -1.0, 0.1, 1.0, 1.0, 0.6, 1, 0)),
            asdict(TrainingMetrics(2, 4, 1.0, 0.1, 0.5, 1.0, 0.35, 1, 1)),
        ])
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8, aggregation="attention")
        (self.root / "initial").mkdir()
        torch.save(dict(model=model.state_dict(), model_config=model.architecture_config(),
                        environment_steps=0, initialization_seed=0),
                   self.root / "initial/mappo_checkpoint.pt")
        with torch.no_grad():
            model.actor_mean[-1].bias.add_(0.1)
        torch.save(dict(model=model.state_dict(), model_config=model.architecture_config(),
                        environment_steps=4, config=asdict(MAPPOConfig(rollout_steps=2, update_epochs=1))),
                   self.root / "training/mappo_checkpoint.pt")
        for stage in ("initial", "final"):
            self._json(self.root / f"evaluation_{stage}/scenario_config.json", [asdict(
                ExperimentScenario("nominal", 3, episodes=2, base_seed=10000, max_steps=2)
            )])
            records = []
            for policy in ("rule", "replacement", "learned"):
                for episode in range(2):
                    trained = stage == "final" and policy == "learned"
                    success = trained and episode == 1
                    reward = (2.0 + episode * 2) if trained else float(episode)
                    records.append(asdict(EpisodeMetrics(
                        scenario="nominal", policy=policy, episode=episode, seed=10000 + episode,
                        agent_count=3, success=success, steps=2,
                        convergence_steps=2 if success else None, cumulative_team_reward=reward,
                        mean_connectivity=1.0, mean_consistency_error=0.1, collisions=0,
                        final_detection_ratio=(0.5 + 0.5 * episode) if trained else 0.25,
                        fault_recovery_time=None, topology_performance_drop=None,
                        replacement_response_time=None, coverage_recovery_time=None,
                        mean_uncovered_lane_ratio=0, replacement_switches=0, replacement_success_rate=None,
                    )))
            self._jsonl(self.root / f"evaluation_{stage}/episode_metrics.jsonl", records)
            self._provenance(stage)

    def _provenance(self, stage):
        """模拟评估器已写出的来源清单；所有统计值仍来自手算测试样例。"""

        directory = self.root / f"evaluation_{stage}"
        checkpoint = self.root / ("initial" if stage == "initial" else "training") / "mappo_checkpoint.pt"
        config = json.loads((self.root / "training/run_config.json").read_text(encoding="utf-8"))
        self._json(directory / "evaluation_provenance.json", dict(
            schema_version=1, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            model_config=config["model"], deterministic=True,
            result_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                           for name in ("scenario_config.json", "episode_metrics.jsonl")},
        ))

    @staticmethod
    def _json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _analyze(self):
        self.assertTrue(SCRIPT.is_file(), "试训分析入口尚未实现")
        return importlib.import_module("scripts.analyze_training_pilot").analyze_pilot(self.root)

    def _edit_records(self, edit):
        path = self.root / "evaluation_final/episode_metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        edit(rows)
        self._jsonl(path, rows)
        self._provenance("final")

    def test_paired_differences_and_baseline_deduplication(self):
        report = self._analyze()
        self.assertEqual(len(report["summaries"]), 4)
        comparison = next(row for row in report["comparisons"] if row["reference"] == "initial")
        # 两个配对收益差为 2 和 3；不是两组均值的相对百分比。
        self.assertEqual(comparison["reward_delta"], 2.5)
        self.assertEqual(comparison["success_rate_delta"], 0.5)
        self.assertEqual(comparison["detection_ratio_delta"], 0.5)
        self.assertEqual(report["training"]["environment_steps"], 4)
        self.assertEqual(report["training"]["episodes_finished"], 2)
        self.assertTrue(report["training"]["parameters_changed"])
        self.assertEqual(report["training_seed_count"], 1)

    def test_rejects_missing_evaluation_episode(self):
        self._edit_records(lambda rows: rows.pop())
        with self.assertRaises(ValueError):
            self._analyze()

    def test_rejects_duplicate_evaluation_episode(self):
        self._edit_records(lambda rows: rows.append(rows[-1]))
        with self.assertRaises(ValueError):
            self._analyze()

    def test_rejects_incomplete_failed_episode(self):
        self._edit_records(lambda rows: rows[-2].update(steps=1, cumulative_team_reward=0))
        with self.assertRaisesRegex(ValueError, "失败回合"):
            self._analyze()

    def test_rejects_wrong_checkpoint_provenance(self):
        path = self.root / "evaluation_final/evaluation_provenance.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["checkpoint_sha256"] = "0" * 64
        self._json(path, value)
        with self.assertRaisesRegex(ValueError, "来源|指纹"):
            self._analyze()

    def test_rejects_changed_results_after_evaluation(self):
        path = self.root / "evaluation_final/episode_metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["cumulative_team_reward"] = 100
        self._jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "来源|指纹"):
            self._analyze()

    def test_rejects_missing_provenance_instead_of_retroactively_certifying(self):
        (self.root / "evaluation_final/evaluation_provenance.json").unlink()
        with self.assertRaises(ValueError):
            self._analyze()

    def test_rejects_evaluation_with_different_motion_semantics(self):
        path = self.root / "evaluation_final/evaluation_provenance.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["motion_profile"] = "displacement_velocity_v2"
        self._json(path, metadata)
        with self.assertRaises(ValueError):
            self._analyze()

    def test_rejects_changed_baseline(self):
        self._edit_records(lambda rows: rows[0].update(cumulative_team_reward=123))
        with self.assertRaisesRegex(ValueError, "基线"):
            self._analyze()

    def test_rejects_mismatched_scenario_protocol(self):
        path = self.root / "evaluation_final/scenario_config.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows[0]["neighbor_state_ttl"] = 2.0
        self._json(path, rows)
        self._provenance("final")
        with self.assertRaises(ValueError):
            self._analyze()

    def test_rejects_training_seed_leakage(self):
        for stage in ("initial", "final"):
            path = self.root / f"evaluation_{stage}/scenario_config.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            rows[0]["base_seed"] = 1
            self._json(path, rows)
            path = path.with_name("episode_metrics.jsonl")
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for record in records:
                record["seed"] = 1 + record["episode"]
            self._jsonl(path, records)
            self._provenance(stage)
        with self.assertRaisesRegex(ValueError, "种子"):
            self._analyze()

    def test_rejects_nonfinite_metric(self):
        self._edit_records(lambda rows: rows[-1].update(cumulative_team_reward=float("nan")))
        with self.assertRaises(ValueError):
            self._analyze()

    def test_rejects_incomplete_training_budget(self):
        path = self.root / "training/training_metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["environment_steps"] = 3
        self._jsonl(path, rows)
        with self.assertRaises(ValueError):
            self._analyze()

    def test_rejects_checkpoint_budget_mismatch(self):
        path = self.root / "training/mappo_checkpoint.pt"
        value = torch.load(path, weights_only=True)
        value["environment_steps"] = 3
        torch.save(value, path)
        with self.assertRaises(ValueError):
            self._analyze()

    def test_absent_recovery_stays_null(self):
        report = self._analyze()
        self.assertTrue(all(row["mean_recovery_time"] is None for row in report["summaries"]))

    def test_cli_outputs_report_and_refuses_overwrite(self):
        self.assertTrue(SCRIPT.is_file(), "试训分析入口尚未实现")
        command = [sys.executable, "-B", "-X", "utf8", str(SCRIPT),
                   "--input-dir", str(self.root), "--output-dir", str(self.root / "analysis")]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "analysis"
        for name in ("pilot_analysis.json", "paired_comparisons.csv", "pilot_comparison.png"):
            self.assertTrue((output / name).is_file())
        source = (output / "pilot_analysis.json").read_bytes()
        repeated = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(source, (output / "pilot_analysis.json").read_bytes())

    def test_evaluator_records_fingerprint_of_loaded_checkpoint_and_results(self):
        directory = self.root / "actual_evaluation"
        checkpoint = self.root / "initial/mappo_checkpoint.pt"
        result = subprocess.run([
            sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/evaluate_policies.py"),
            "--checkpoint", str(checkpoint), "--episodes", "1", "--agent-counts", "2",
            "--max-steps", "2", "--output-dir", str(directory),
        ], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        path = directory / "evaluation_provenance.json"
        self.assertTrue(path.is_file(), "评估执行时尚未记录检查点来源")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        for name in ("scenario_config.json", "episode_metrics.jsonl"):
            self.assertEqual(metadata["result_sha256"][name], hashlib.sha256((directory / name).read_bytes()).hexdigest())

    def test_analysis_accepts_explicit_new_evaluation_directories(self):
        self.assertTrue(SCRIPT.is_file(), "试训分析入口尚未实现")
        # 移动的仅为临时测试副本，验证不必覆盖旧实验即可核查重跑评估。
        initial = self.root / "evaluation_initial_verified"
        final = self.root / "evaluation_final_verified"
        (self.root / "evaluation_initial").rename(initial)
        (self.root / "evaluation_final").rename(final)
        result = subprocess.run([
            sys.executable, "-B", "-X", "utf8", str(SCRIPT), "--input-dir", str(self.root),
            "--initial-evaluation", str(initial), "--final-evaluation", str(final),
            "--output-dir", str(self.root / "analysis_verified"),
        ], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "analysis_verified/pilot_analysis.json").is_file())


if __name__ == "__main__":
    unittest.main()

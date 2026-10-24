"""用真实小规模训练验证阶段级续跑、原始结果保护及并发锁。"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AblationResumeTests(unittest.TestCase):
    @classmethod
    def command(cls, directory, *extra):
        return [
            sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/run_ablation.py"),
            "--aggregations", "mean", "--seeds", "0", "--train-agents", "2",
            "--agent-counts", "2", "--updates", "1", "--rollout-steps", "4",
            "--update-epochs", "1", "--max-steps", "4", "--hidden-dim", "8",
            "--episodes", "1", "--output-dir", str(directory), *extra,
        ]

    @classmethod
    def setUpClass(cls):
        cls.template_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.template_directory.cleanup)
        cls.template = Path(cls.template_directory.name) / "complete_experiment"
        result = subprocess.run(cls.command(cls.template), cwd=PROJECT_ROOT,
                                capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"最小真实实验失败: {result.stdout}\n{result.stderr}")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "copied_experiment"
        shutil.copytree(self.template, self.root)
        self.run_dir = self.root / "mean/seed_0"

    @staticmethod
    def hashes(directory):
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file()}

    def invoke(self, *extra):
        return subprocess.run(self.command(self.root, *extra), cwd=PROJECT_ROOT,
                              capture_output=True, text=True, encoding="utf-8")

    def test_resume_reuses_valid_training_and_evaluation_after_directory_move(self):
        before = {stage: self.hashes(self.run_dir / stage) for stage in ("training", "evaluation")}
        result = self.invoke("--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("跳过训练", result.stdout)
        self.assertIn("跳过评估", result.stdout)
        for stage in before:
            self.assertEqual(self.hashes(self.run_dir / stage), before[stage])
        report = json.loads((self.root / "summary/ablation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["completed_runs"], 1)
        self.assertEqual(report["source_directory"], str(self.root.resolve()))

    def test_resume_rejects_changed_budget_before_touching_stage_outputs(self):
        before = self.hashes(self.run_dir)
        result = self.invoke("--resume", "--updates", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("计划", result.stderr)
        self.assertEqual(self.hashes(self.run_dir), before)

    def test_incomplete_evaluation_is_archived_and_retried_without_retraining(self):
        source = self.run_dir / "evaluation/episode_metrics.jsonl"
        lines = source.read_text(encoding="utf-8").splitlines()
        source.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        incomplete = self.hashes(self.run_dir / "evaluation")
        training = self.hashes(self.run_dir / "training")
        # 原计划只当配置读取；即使旧程序路径失效，也应执行当前项目的评估入口。
        plan_path = self.root / "ablation_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan[0]["evaluate"][:2] = ["missing-old-python", "missing-old-evaluation.py"]
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        result = self.invoke("--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.hashes(self.run_dir / "training"), training)
        archives = list((self.run_dir / "attempts").glob("evaluation-*"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(self.hashes(archives[0]), incomplete)
        self.assertEqual(len(source.read_text(encoding="utf-8").splitlines()), 9)

    def test_corrupt_checkpoint_archives_training_and_invalidates_old_evaluation(self):
        checkpoint = self.run_dir / "training/mappo_checkpoint.pt"
        checkpoint.write_bytes(b"interrupted-checkpoint")
        training = self.hashes(self.run_dir / "training")
        evaluation = self.hashes(self.run_dir / "evaluation")
        result = self.invoke("--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        for stage, original in (("training", training), ("evaluation", evaluation)):
            archives = list((self.run_dir / "attempts").glob(stage + "-*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(self.hashes(archives[0]), original)
        self.assertNotEqual(checkpoint.read_bytes(), b"interrupted-checkpoint")

    def test_partial_summary_is_preserved_before_regeneration(self):
        partial = b'{"interrupted":'
        (self.root / "summary/ablation_summary.json").write_bytes(partial)
        before = {stage: self.hashes(self.run_dir / stage) for stage in ("training", "evaluation")}
        result = self.invoke("--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        archived = list((self.root / "attempts").glob("summary-*"))
        self.assertEqual(len(archived), 1)
        self.assertEqual((archived[0] / "ablation_summary.json").read_bytes(), partial)
        for stage in before:
            self.assertEqual(self.hashes(self.run_dir / stage), before[stage])

    def test_resume_preview_does_not_change_existing_experiment(self):
        before = self.hashes(self.root)
        result = self.invoke("--resume", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.hashes(self.root), before)

    def test_partial_csv_and_png_are_not_reused_as_complete_summary(self):
        import csv
        from PIL import Image

        relocated = self.invoke("--resume")
        self.assertEqual(relocated.returncode, 0, relocated.stderr)
        raw = {stage: self.hashes(self.run_dir / stage) for stage in ("training", "evaluation")}
        for name, partial in (("ablation_summary.csv", b"aggregation,policy\n"),
                              ("ablation_comparison.png", b"\x89PNG\r\n\x1a\n")):
            with self.subTest(name=name):
                target = self.root / "summary" / name
                target.write_bytes(partial)
                result = self.invoke("--resume")
                self.assertEqual(result.returncode, 0, result.stderr)
                if name.endswith("csv"):
                    with target.open(encoding="utf-8-sig", newline="") as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), 9)
                else:
                    self.assertNotEqual(target.read_bytes(), partial)
                    with Image.open(target) as image:
                        image.verify()
                for stage in raw:
                    self.assertEqual(self.hashes(self.run_dir / stage), raw[stage])

    def test_archive_never_targets_the_experiment_root(self):
        from scripts.run_ablation import _archive

        with self.assertRaises(ValueError):
            _archive(self.root, self.root)
        self.assertFalse((self.root.parent / "attempts").exists())

    def test_existing_results_still_require_explicit_resume(self):
        before = self.hashes(self.run_dir)
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("已有", result.stderr)
        self.assertEqual(self.hashes(self.run_dir), before)

    def test_resume_requires_original_plan(self):
        (self.root / "ablation_plan.json").unlink()
        before = self.hashes(self.run_dir)
        result = self.invoke("--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("计划", result.stderr)
        self.assertEqual(self.hashes(self.run_dir), before)

    def test_experiment_lock_prevents_concurrent_writer_and_releases_after_exit(self):
        from scripts.run_ablation import experiment_lock

        with experiment_lock(self.root):
            blocked = self.invoke("--resume", "--dry-run")
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("运行", blocked.stderr)
        after = self.invoke("--resume", "--dry-run")
        self.assertEqual(after.returncode, 0, after.stderr)


if __name__ == "__main__":
    unittest.main()

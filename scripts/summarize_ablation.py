"""独立汇总已有消融实验；不启动训练，也不补造缺失实验。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ablation import save_ablation_report, summarize_ablation


def main() -> None:
    """读取计划及实际配置，校验成功后输出跨训练种子的统计表和图。"""

    parser = argparse.ArgumentParser(description="检查并汇总多种子消融实验")
    parser.add_argument("--input-dir", required=True, type=Path, help="包含 ablation_plan.json 的实验目录")
    parser.add_argument("--output-dir", type=Path, help="必须为新目录，默认使用 input-dir/summary")
    args = parser.parse_args()
    try:
        report = summarize_ablation(args.input_dir)
        paths = save_ablation_report(report, args.output_dir or args.input_dir / "summary")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"汇总失败: {exc}\n")
    print(f"已核验 {report['completed_runs']} 个独立训练组")
    for warning in report["warnings"]:
        print(f"统计提示: {warning}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()

"""CLI for the frozen TD-13 standalone evaluator."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from mvp_evaluation import evaluate_checkpoint, publish_evaluation


ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mvp_evaluation.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    summary, samples = evaluate_checkpoint(
        repository_root=ROOT,
        evaluation_config_path=args.config.resolve(),
        checkpoint=args.checkpoint,
        split=args.split,
        allow_test=args.allow_test,
    )
    manifest = publish_evaluation(args.output_directory.resolve(), summary, samples)
    result = {**summary, "publication": {"path": str(args.output_directory.resolve()), "manifest": manifest}}
    if args.report:
        _atomic_json(args.report.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

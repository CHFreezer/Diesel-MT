"""Run the locked Hy-MT2 7B teacher in a network-blocked offline process."""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

from hymt2_teacher_runtime import (
    atomic_write_json,
    file_identity,
    inspect_snapshot,
    load_json_mapping,
    load_yaml_mapping,
    resolve_runtime_root,
    run_offline_smoke,
    runtime_paths,
    validate_contract,
    validate_environment,
    verify_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/hymt2_teacher_runtime.yaml")
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/hymt2_teacher_artifact.lock.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    profile = load_yaml_mapping(args.profile)
    lock = load_json_mapping(args.lock)
    validate_contract(profile, lock)
    root = resolve_runtime_root(profile, ROOT)
    paths = runtime_paths(profile, root)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    output = paths["reports"] / ("runtime-verification.json" if args.verify_only else "offline-smoke.json")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "artifact": {
            "repo_id": lock["selected"]["repo_id"],
            "revision": lock["selected"]["revision"],
            "profile": file_identity(args.profile.resolve()),
            "lock": file_identity(args.lock.resolve()),
        },
        "runtime_root": str(root),
    }
    try:
        report["environment"] = validate_environment(profile)
        report["verified_files"] = verify_snapshot(paths["snapshot"], lock)
        report["inspection"] = inspect_snapshot(paths["snapshot"], lock)
        if args.verify_only:
            report["status"] = "verified"
        else:
            report["smoke"] = run_offline_smoke(paths["snapshot"], profile)
            report["status"] = "pass"
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        atomic_write_json(output, report)
        raise
    atomic_write_json(output, report)
    print(f"teacher runtime status: {report['status']}")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare and verify the locked Hy-MT2 7B teacher artifact and overlay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hymt2_teacher_runtime import (
    atomic_write_json,
    create_overlay,
    download_locked_snapshot,
    file_identity,
    inspect_snapshot,
    load_json_mapping,
    load_yaml_mapping,
    resolve_runtime_root,
    runtime_paths,
    validate_contract,
    verify_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/hymt2_teacher_runtime.yaml")
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/hymt2_teacher_artifact.lock.json")
    parser.add_argument("--create-overlay", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    profile = load_yaml_mapping(args.profile)
    lock = load_json_mapping(args.lock)
    validate_contract(profile, lock)
    root = resolve_runtime_root(profile, ROOT)
    paths = runtime_paths(profile, root)
    for path in (paths["root"], paths["reports"]):
        path.mkdir(parents=True, exist_ok=True)

    if args.create_overlay:
        variables = {**os.environ, **{str(k): str(v) for k, v in profile["environment"]["variables"].items()}}
        create_overlay(
            paths["overlay"],
            ROOT / ".conda/python.exe",
            ROOT / str(profile["environment"]["requirements"]),
            env=variables,
        )
    if args.download:
        download_locked_snapshot(paths["snapshot"], lock, max_workers=args.max_workers)
    if not (args.download or args.verify_only):
        parser.error("select --download or --verify-only (optionally with --create-overlay)")

    verified = verify_snapshot(paths["snapshot"], lock)
    inspection = inspect_snapshot(paths["snapshot"], lock)
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "repo_id": lock["selected"]["repo_id"],
        "revision": lock["selected"]["revision"],
        "snapshot": str(paths["snapshot"]),
        "profile": file_identity(args.profile.resolve()),
        "lock": file_identity(args.lock.resolve()),
        "files": verified,
        "inspection": inspection,
    }
    output = paths["reports"] / "artifact-verification.json"
    atomic_write_json(output, manifest)
    print(f"verified {len(verified)} locked files ({lock['selected']['total_bytes']} bytes)")
    print(f"runtime root: {root}")
    print(f"verification report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

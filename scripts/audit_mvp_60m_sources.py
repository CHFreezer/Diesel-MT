"""Audit and select quality-actual native Traditional Chinese source text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mvp_60m_data_pipeline import (
    apply_domain_ceilings,
    audit_candidates,
    iter_hant_candidates,
    sha256_file,
    write_json,
    write_jsonl,
)
from tokenizer_utils import reload_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--massive", type=Path, required=True)
    parser.add_argument("--moj", type=Path, nargs=2, required=True)
    parser.add_argument("--hkel-hant", type=Path, required=True)
    parser.add_argument("--mdn", type=Path, required=True)
    parser.add_argument("--tldr", type=Path, required=True)
    parser.add_argument("--ud-hk", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    runtime_root = args.runtime_root.resolve()
    inputs = {
        "massive": args.massive.resolve(),
        "moj": [path.resolve() for path in args.moj],
        "hkel_hant": args.hkel_hant.resolve(),
        "mdn": args.mdn.resolve(),
        "tldr": args.tldr.resolve(),
        "ud_hk": args.ud_hk.resolve(),
    }
    for value in inputs.values():
        paths = value if isinstance(value, list) else [value]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    tokenizer = reload_tokenizer(repository_root / "artifacts/tokenizers/mvp-tokenizer-v0")
    flores_dev: list[str] = []
    for language in ("eng_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang"):
        path = repository_root / "data/model/raw/flores200-original/dev" / f"{language}.dev"
        flores_dev.extend(path.read_text(encoding="utf-8").splitlines())

    accepted, report = audit_candidates(
        iter_hant_candidates(inputs), tokenizer=tokenizer, contamination_texts=flores_dev
    )
    selected = apply_domain_ceilings(accepted)
    domain_counts = Counter(str(record["domain"]) for record in selected)
    source_counts = Counter(str(record["source_id"]) for record in selected)
    selection_path = runtime_root / "td02" / "native-hant-selected.jsonl"
    records, digest = write_jsonl(selection_path, selected)
    report.update(
        {
            "status": "complete",
            "formal_test_accessed": False,
            "selection_policy": "quality-actual; no quota fill; technical<=15%; legal/government<=20%",
            "selected_records": records,
            "selected_sha256": digest,
            "selected_domain_counts": dict(sorted(domain_counts.items())),
            "selected_source_counts": dict(sorted(source_counts.items())),
            "selected_domain_shares": {
                key: count / records for key, count in sorted(domain_counts.items())
            },
            "input_files": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in [
                    inputs["massive"],
                    *inputs["moj"],
                    inputs["hkel_hant"],
                    inputs["mdn"],
                    inputs["tldr"],
                    inputs["ud_hk"],
                ]
            ],
        }
    )
    write_json(runtime_root / "td02" / "source-audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

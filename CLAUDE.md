# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Diesel-MT is a lightweight multilingual machine translation experiment targeting Chinese, English, Japanese, and Korean (12 directed translation pairs). It uses an M2M100-style Encoder-Decoder Transformer trained from scratch, with a baseline target of ~200M parameters for CPU/mobile SoC deployment. Hy-MT2 7B serves as an offline distillation teacher.

## Development environment

- **Python**: 3.11.15 in a project-local `.conda/` prefix (not a named environment)
- **Shell**: PowerShell 7.6 (`pwsh.exe`)
- **Package manager**: pip only (no `pyproject.toml` or conda packages); always use `python -m pip`, never bare `pip`
- **Platform**: Windows 11 Pro, E: spinning disk for data, D: NVMe SSD

Activate the environment in every new PowerShell session:
```pwsh
& 'C:\Users\chfre\miniconda3\shell\condabin\conda-hook.ps1'
conda activate (Join-Path $PWD '.conda')
```

Install dependencies:
```pwsh
python -m pip install -r requirements.txt
```

## Commands

```bash
# Run all tests (from repo root)
.conda\python.exe -m pytest -q

# Run a single test
.conda\python.exe -m pytest tests/test_tokenizer_dataset_pipeline.py -k test_name

# Estimate model parameters for all configurations
python scripts/calculate_model_parameters.py

# Fetch tokenizer datasets — dry-run first
python scripts/fetch_tokenizer_datasets.py --profile smoke --dry-run

# Fetch tokenizer datasets — actual run
python scripts/fetch_tokenizer_datasets.py --profile mvp [--resume]

# Resolve/update source lock (requires network)
python scripts/fetch_tokenizer_datasets.py --resolve-lock
```

## Architecture

### No package structure

The project has no `src/` layout, `__init__.py` files, or installable package. Python modules live in `scripts/` as flat files. Tests import them via `sys.path.insert(0, str(ROOT / "scripts"))`, so in tests the scripts directory acts as an ad-hoc package.

### Key modules

- **`scripts/fetch_tokenizer_datasets.py`** — CLI entry point for the tokenizer corpus pipeline. Thin argument parsing + delegation to the pipeline library.
- **`scripts/tokenizer_dataset_pipeline.py`** (~1543 lines) — Core processing library: config validation, HPLT 3.0 HTTP fetcher with range/resume, text cleaning pipeline, MinHash approximate dedup, deterministic balanced sampling, memory-first builds, per-language checkpointing, atomic file output, quality reports.
- **`scripts/calculate_model_parameters.py`** — Standalone parameter estimator for 5 model configs (baseline + 4 MVP candidates).

### Config and lock system

- `configs/tokenizer_datasets_mvp.yaml` — Source registry, cleaning rules, MinHash params, quality thresholds, `smoke` and `mvp` profiles.
- `configs/tokenizer_datasets_mvp.lock.json` — Pinned HPLT 3.0 shard URLs, SHA-256 hashes, byte ranges for deterministic reproducibility. The lock binds to a config hash — if config or profile changes, the lock must be re-resolved.

### Data flow

```
config + lock → download (HPLT .jsonl.zst shards) → text extraction + cleaning
→ MinHash dedup fingerprint → balanced language sampling → corpus .txt files + manifest.jsonl → quality report
```

Output layout under `data/tokenizer/` (all gitignored except `.gitkeep`): `raw/` (downloaded shards), `cache/`, `interim/` (per-language checkpoint state), `corpus/` (final `.txt` files + `manifest.jsonl`), `reports/`.

### Design invariants

1. **Deterministic reproducibility**: byte-level determinism via seeded algorithms, locked sources, and config hashing. Same inputs must produce identical outputs.
2. **Memory-first**: optimized for single-language in-RAM processing on 128 GB; fingerprints via `ProcessPoolExecutor`, decisions serial in main process.
3. **Conservative cleaning**: never lowercases, normalizes scripts, or does character-set folding. Only removes content-invalid lines (garbled characters, wrong-script dominance).
4. **Atomic output**: all files written via temp file + `os.replace()`. Manifest is written last, after all content is verified.
5. **Per-language checkpointing**: resume at language granularity; identity hash = config + lock + seed + code version.

## Project workflow

```
work/plan/    → work/todo/    → work/task/    → work/review/    → work/done/
```

Current state:
- **Completed**: Tokenizer dataset fetch pipeline (TD-01 through TD-12, fully tested, MVP corpus built with ~4B characters across 4 languages).
- **Next phase**: MVP tokenizer training (`work/plan/mvp-tokenizer.md`). Train a SentencePiece-compatible tokenizer from scratch with 32k and 48k vocab sizes, using the corpus in `data/tokenizer/corpus/mvp/`.

## Testing

Tests in `tests/test_tokenizer_dataset_pipeline.py` (19 tests, 526 lines). They simulate HPLT sources with in-memory fixtures — no network dependency. Key patterns:

- Config validation (explicit registry, missing fields, error paths)
- Text cleaning correctness (zh/ja/ko-specific patterns)
- MinHash fingerprint stability and similarity
- Deterministic build reproducibility (byte-level across two runs)
- Checkpoint resume behavior
- Cache validation and network-failure handling
- CLI dry-run output checks
- Atomic-output guarantee (no half-written manifest)

Fixtures in `tests/fixtures/tokenizer_datasets/` are small JSONL samples for each language.

## Model configuration

Target baseline (M2M100Config semantics): vocab=64k, d_model=768, encoder_layers=16, decoder_layers=4, ffn_dim=3072, attention_heads=12, tie_word_embeddings=true → ~201M params.

MVP rapid-validation configs range from 50M–75M params using d_model=512, ffn_dim=2048, with `e12-d3` or `e8-d2` layer counts and 32k/48k vocab options. See `scripts/calculate_model_parameters.py` for all presets.

# AGENTS.md

This file provides repository guidance to coding agents working in this project.

## Project overview

Diesel-MT is a lightweight multilingual machine translation experiment targeting four product languages: Chinese, English, Japanese, and Korean. They form 12 directed cross-language product translation directions. Chinese uses two independently selectable model-facing tags, `zho_Hans` for Simplified Chinese and `zho_Hant` for Traditional Chinese. The frozen tokenizer therefore has five language tags, while the complete model capability matrix has 20 directed routes: 18 cross-language translation routes plus two Simplified/Traditional Chinese conversion routes. The model uses an M2M100-style Encoder-Decoder Transformer trained from scratch, with a baseline target of ~200M parameters for CPU/mobile SoC deployment. The frozen offline distillation teacher is the official Hy-MT2 7B GGUF Q8_0 artifact running through pinned llama.cpp CUDA.

## Development environment

- **Python**: 3.11.15 in a project-local `.conda/` prefix (not a named environment)
- **Shell**: PowerShell 7.6 (`pwsh.exe`)
- **Package manager**: pip only (no `pyproject.toml` or conda packages); always use `python -m pip`, never bare `pip`
- **Platform**: Windows 11 Pro. Machine-specific CPU/GPU, memory, drive mapping, and measured execution envelopes live only in the optional Git-excluded root `LOCAL_HARDWARE.md`.

Activate the environment in every new PowerShell session:
```pwsh
& 'C:\Users\chfre\miniconda3\shell\condabin\conda-hook.ps1'
conda activate (Join-Path $PWD '.conda')
```

Install dependencies:
```pwsh
python -m pip install -r requirements.txt
```

### Local hardware boundary

- `LOCAL_HARDWARE.md` is the single local hardware/execution record. It is excluded through `.git/info/exclude` and must never be committed.
- Do not copy GPU model names, fixed VRAM/RAM sizes, drive letters, or host-specific worker counts into reusable task titles or implementation branches.
- Training code must consume configurable resource budgets and probe the current runtime. The run manifest records actual device, driver/backend, memory, and storage roots; changing hardware changes the resource profile, not the algorithm.

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

### Language and direction terminology

- **Product languages**: 4 — Chinese, English, Japanese, Korean.
- **Model language tags**: 5 — `zho_Hans`, `zho_Hant`, `eng_Latn`, `jpn_Jpan`, `kor_Hang`.
- **Cross-language product translation directions**: 12.
- **Model relation groups**: 10 undirected tag pairs — nine cross-language translation relations plus one `zho_Hans--zho_Hant` Chinese-internal conversion relation.
- **Model routes**: 20 — 18 cross-language translation routes plus `zho_Hans -> zho_Hant` and `zho_Hant -> zho_Hans`.
- Keep the existing teacher language names: `zho_Hans -> Chinese` and `zho_Hant -> Traditional Chinese`; do not introduce locale-specific prompt names for this amendment.
- Use “Chinese” only for a product-level statement that applies to both Chinese states. Data, configs, training, inference, and metrics must name `zho_Hans` or `zho_Hant` explicitly. Aggregated Chinese metrics must retain both tag-level route breakdowns.
- The transition contract and immutable-v1 boundary are frozen in `docs/chinese-locale-capability-contract.md`.

### No package structure

The project has no `src/` layout, `__init__.py` files, or installable package. Python modules live in `scripts/` as flat files. Tests import them via `sys.path.insert(0, str(ROOT / "scripts"))`, so in tests the scripts directory acts as an ad-hoc package.

### Key modules

- **`scripts/fetch_tokenizer_datasets.py`** — CLI entry point for the tokenizer corpus pipeline. Thin argument parsing + delegation to the pipeline library.
- **`scripts/tokenizer_dataset_pipeline.py`** (~1543 lines) — Core processing library: config validation, HPLT 3.0 HTTP fetcher with range/resume, text cleaning pipeline, MinHash approximate dedup, deterministic balanced sampling, memory-first builds, per-language checkpointing, atomic file output, quality reports.
- **`scripts/calculate_model_parameters.py`** — Standalone parameter estimator for 5 model configs (baseline + 4 MVP candidates).
- **`scripts/model_training_contract.py`** — Strict MVP model-data/student/source-lock contract: five tags, ten undirected relations, 20 routes, canonical config hashing, provenance schema, path boundaries, and fail-fast validation.
- **`scripts/prepare_model_data.py`** — Thin TD-03 CLI for side-effect-free dry runs, locked-cache/offline builds, and identity-bound locale checkpoint resume.
- **`scripts/model_data_pipeline.py`** — Deterministic MASSIVE parallel-data adapter: resumable archive fetch, nested file verification, conservative multilingual cleaning, stable sample/group identities, provenance, atomic corpus/report publication, and manifest-last completion.
- **`scripts/finalize_model_data.py`** — Thin TD-04 CLI that validates the TD-03 manifest, enforces external-reference completeness, and publishes finalized split data only after contamination checks.
- **`scripts/model_data_split_pipeline.py`** — Group/component hash split, exact/near deduplication, derivation binding, external tokenizer/evaluation contamination scan, reverse-route expansion, test-identity freeze, leakage audit, and manifest-last publication.
- **`scripts/hymt2_teacher_runtime.py`** — TD-06 teacher contract, artifact/hash verification, isolated overlay audit, socket-blocked five-tag inference, M0 capacity probe, and runtime/resource evidence.
- **`scripts/prepare_hymt2_teacher.py`** / **`scripts/validate_hymt2_teacher_runtime.py`** — Thin CLIs for locked teacher snapshot preparation and fully offline validation.
- **`scripts/benchmark_hymt2_teacher_variants.py`** — TD-06 common-protocol benchmark for original unquantized BF16, bitsandbytes LLM.int8, and official GGUF Q8_0 + llama.cpp CUDA, including full artifact verification, BF16-based output comparison, and 200 ms RAM/VRAM sampling.
- **`scripts/hymt2_distillation.py`** — Shared TD-07/TD-08 prompt, filtering, metrics, deterministic sampling, and loopback llama.cpp teacher runtime contracts.
- **`scripts/calibrate_hymt2_teacher.py`** — TD-07 dev-only prompt/decode calibration and exact-replay CLI; calibration outputs never enter student training.
- **`scripts/hymt2_distillation_data.py`** / **`scripts/generate_teacher_data.py`** — TD-08 train-only bounded sequence-distillation pipeline and CLI: deterministic route sampling, per-sample resume, raw/accepted/filtered separation, manual review, replay, gates, and manifest-last publication.
- **`scripts/mvp_student.py`** / **`scripts/mvp_training.py`** / **`scripts/mvp_checkpoint.py`** — TD-09 through TD-11 frozen-tokenizer student construction, direction-aware encoding, deterministic 20-route sampling/training, explicit resource budgets, and atomic exact-resume checkpoints.
- **`scripts/mvp_evaluation.py`** / **`scripts/evaluate_mvp_model.py`** — TD-13 standalone dev/test-gated evaluator with fixed SacreBLEU/chrF semantics, 20 tag routes, 12 product-direction aggregates, and two Chinese conversion routes.
- **`scripts/mvp_resource_benchmark.py`** / **`scripts/mvp_resource_profile.py`** — TD-14 real-length resource candidates, runtime hardware evidence, unique M2 profile selection, 100-step soak, and resume acceptance.
- **`scripts/mvp_distillation_ab.py`** — TD-15 strict accepted-intersection builder and fairness validator for source-identical human-only/distilled recipes, pretraining differences, paired dry runs, and frozen dev selection rules.
- **`scripts/build_micro_m2m100_checkpoint.py`** — Deterministically builds and validates the Git-ignored random HF checkpoint consumed by the CTranslate2 deployment workflow.
- **`scripts/validate_ctranslate2_deployment.py`** — Runs the serial CT2 conversion, ordered-vocabulary validation, five-tag CPU inference, and offline-package phases while merging machine-readable results into one workflow JSON.
- **`scripts/run_offline_ctranslate2_smoke.py`** — Self-contained deployment-root runner copied into the offline package; verifies its manifest and blocks Python socket connections before local inference.

### Config and lock system

- `configs/tokenizer_datasets_mvp.yaml` — Source registry, cleaning rules, MinHash params, quality thresholds, `smoke` and `mvp` profiles.
- `configs/tokenizer_datasets_mvp.lock.json` — Pinned HPLT 3.0 shard URLs, SHA-256 hashes, byte ranges for deterministic reproducibility. The lock binds to a config hash — if config or profile changes, the lock must be re-resolved.
- `configs/mvp_model_data.yaml` / `.lock.json` — Schema v2 identities for the ten-relation/20-route MASSIVE build, including `zho_Hans--zho_Hant` from the locked `zh-CN`/`zh-TW` source files. Old runtime manifests remain immutable.
- `configs/mvp_e8_d2_v48k.yaml` — From-scratch student identity, logical runtime/publish paths, runtime-probed device/precision preferences, and an explicit configurable resource-budget schema. The canonical base keeps candidate values null; TD-10/TD-12/TD-14 candidate profiles may fill all budget fields, and TD-14 freezes the selected profile.
- `configs/hymt2_teacher_selection.yaml` — Canonical frozen TD-06 selection: official Hy-MT2 7B GGUF Q8_0 through pinned llama.cpp CUDA. Original unquantized BF16 is the quantization-quality baseline. Both reside under Git-ignored `artifacts/model-training/runtime/` and are read-mostly/sequential-load assets. TD-07/TD-08 must consume the selected identity and may not silently fall back to another backend.
- `configs/hymt2_teacher_runtime.yaml` / `hymt2_teacher_artifact.lock.json` — Non-selected FP8 baseline profile and immutable evidence. The validated native-Windows path decompresses to BF16 and is retained only for audit/comparison.
- `configs/hymt2_teacher_benchmark.yaml` / `.lock.json` — Common five-tag benchmark contract plus byte-exact identities for official Hy-MT2 7B BF16, official GGUF Q8_0, and the pinned llama.cpp CUDA runtime.
- `configs/hymt2_teacher_prompt_decode.yaml` / `hymt2_distillation.yaml` / `hymt2_distillation_d1.yaml` — Frozen 18-route v1 TD-07/TD-08 identities. The `*_zh_conversion.yaml` configs and `hymt2_distillation_d1_20route_composite.yaml` freeze the completed two-route addendum and 20-route D1 identity without mutating v1.

### Data flow

```
config + lock → download (HPLT .jsonl.zst shards) → text extraction + cleaning
→ MinHash dedup fingerprint → balanced language sampling → corpus .txt files + manifest.jsonl → quality report
```

Output layout under `data/tokenizer/` (all gitignored except `.gitkeep`): `raw/` (downloaded shards), `cache/`, `interim/` (per-language checkpoint state), `corpus/` (final `.txt` files + `manifest.jsonl`), `reports/`.

### Design invariants

1. **Deterministic reproducibility**: byte-level determinism via seeded algorithms, locked sources, and config hashing. Same inputs must produce identical outputs.
2. **Memory-first with explicit budgets**: single-language candidates stay in RAM; fingerprints use `ProcessPoolExecutor` and decisions remain serial in the main process. Memory, worker, and staging limits come from configuration/local execution profiles rather than assumed host capacity.
3. **Conservative cleaning**: never lowercases, normalizes scripts, or does character-set folding. Only removes content-invalid lines (garbled characters, wrong-script dominance).
4. **Atomic output**: all files written via temp file + `os.replace()`. Manifest is written last, after all content is verified.
5. **Per-language checkpointing**: resume at language granularity; identity hash = config + lock + seed + code version.

## Project workflow

```
work/plan/    → work/todo/    → work/task/    → work/review/    → work/done/
```

Current state:
- **Completed**: Tokenizer dataset fetch pipeline (TD-01 through TD-12), the bounded MVP tokenizer workflow, and CTranslate2 deployment validation. The frozen `mvp-tokenizer-v0` is a 49,152-token Hugging Face Rust BPE + Metaspace artifact for five model tags: `eng_Latn`, `zho_Hans`, `zho_Hant`, `jpn_Jpan`, and `kor_Hang`.
- **Archived workflows**: Plans remain under `work/plan/`; completed todos, task sets, and review records are under `work/done/`. Narrative evidence belongs in the task and unified review documents; the CT2 workflow's single machine-readable record is `artifacts/ctranslate2/deployment-validation.json`.
- **Model-training storage**: The current 20-route corpus is published under `data/model/corpus/mvp/`; the immutable 18-route M0 dependency is explicit under `data/model/history/m0-v1/`. Compact release evidence is grouped by domain under `artifacts/model-training/reports/{m0,teacher,student,m2}/`, while models, environments, checkpoints, and generated benchmark inputs live under `artifacts/model-training/runtime/`. Do not use `td-xx` task prefixes for data or artifact filenames.
- **Active workflow**: `work/plan/mvp-model-training.md`, `work/todo/mvp-model-training.md`, and `work/task/mvp-model-training/`. TD-01 through TD-15 are completed; TD-16 through TD-18 remain pending. The 20-route M0 contains 226,218 human train records, the frozen D1 composite contains 44,361 accepted train-only teacher targets, and the TD-15 source-matched A/B cohort contains 44,313 records. TD-14 froze `configs/mvp_training_m2_profile.yaml`; TD-15 froze the human-only/distilled recipes and dev selection rules without accessing test. M0 v1, D0 v1, and D1 v1 remain immutable component evidence. The frozen tokenizer already has both Chinese tags and must not be rebuilt. Do not reinterpret the random deployment checkpoint or M1 memorization checkpoint as a trained translation model.

## Testing

The offline suite currently collects 192 tests, all passing on the current Windows host. Link rejection is covered without administrator privileges by a real NTFS directory junction/reparse point plus a directed payload-link validator test. The suite spans the tokenizer/model-data pipelines, teacher runtime/calibration/distillation and runtime benchmarks, tokenizer training/checkpointing, standalone model evaluation, M1, resource-profile soak, A/B fairness, artifact-freeze, micro-checkpoint, and CTranslate2 deployment modules. Small fixtures simulate HPLT, MASSIVE 1.1, group split/dedup/leakage, M0/D0/D1 route and acceptance evidence, teacher artifact/offline boundaries, and model-training contracts without network access. Key patterns:

- Config validation (explicit registry, missing fields, error paths)
- Text cleaning correctness (zh/ja/ko-specific patterns)
- MinHash fingerprint stability and similarity
- Deterministic build reproducibility (byte-level across two runs)
- Checkpoint resume behavior
- Cache validation and network-failure handling
- CLI dry-run output checks
- Atomic-output guarantee (no half-written manifest)
- Save/reload and deterministic tokenizer training
- Fixed evaluation-set construction and unknown-character accounting
- Frozen artifact manifest integrity and five-tag micro-M2M100 forwards
- Locked MASSIVE archive/member verification, five-locale alignment, conservative cleaning, stable parallel sample/group IDs, offline/cache/resume reproducibility, and manifest-last failure safety
- Stable component-hash train/dev/test split, exact/near duplicate binding, forward/reverse isolation, derived-group validation, tokenizer/evaluation reference policies, blocked-contamination reports, frozen test identity, and order-independent finalized manifests
- Model-training schema/provenance, 10-pair/20-route invariants, config/source-lock identity, and path-boundary rejection

Fixtures in `tests/fixtures/tokenizer_datasets/` are small JSONL samples for all five model language tags.

## Model configuration

Target baseline (M2M100Config semantics): vocab=64k, d_model=768, encoder_layers=16, decoder_layers=4, ffn_dim=3072, attention_heads=12, tie_word_embeddings=true → ~201M params.

MVP rapid-validation configs range from 50M–75M params using d_model=512, ffn_dim=2048, with `e12-d3` or `e8-d2` layer counts and 32k/48k vocab options. See `scripts/calculate_model_parameters.py` for all presets.

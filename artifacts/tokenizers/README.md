# Diesel-MT tokenizer artifacts

## Frozen MVP artifact

`mvp-tokenizer-v0/` is the frozen five-language tokenizer for
`eng_Latn`, `zho_Hans`, `zho_Hant`, `jpn_Jpan`, and `kor_Hang`.

- Algorithm: Hugging Face Rust BPE + Metaspace, `byte_fallback=false`.
- Final vocabulary size: 49,152, including special and language tokens.
- Training data: five balanced HPLT 3.0 corpora, about 200M cleaned Unicode
  characters per language; `cmn_Hant` is mapped directly to `zho_Hant`.
- Training corpus manifest SHA-256:
  `b3d8d6f4f559813929c75086e6060b74a922a87cdb06646973d1168b5618c977`.
- Independent holdout manifest SHA-256:
  `c5bec116578ea88d37f325c3e18c66a889ef34aa263bb876e821456c500f9ffe`.
- Artifact manifest SHA-256 (freeze root):
  `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`.
- `tokenizer.json` SHA-256:
  `22bceccab939afe1003d1fbdd95d9d4e78eb954b2e9003d70131155666b1201c`.

The artifact contains the tokenizer/configuration, language-token mapping,
training provenance, alphabet audit, corpus manifest snapshot, and a manifest
that hashes every published file. The fixed evaluation set and reports are in
`reports/mvp-tokenizer-v0/`; `freeze_acceptance.md` is the human-readable
release record.

## Engineering fallback

`50pct/mvp-32k/` is retained as the pre-existing 32k four-language engineering
fallback and downstream throughput/parameter-count control. It was not
retrained during the five-language freeze and is not equivalent to the frozen
48k artifact. The comparison report explicitly preserves this provenance
difference.

The older `50pct/mvp-48k/` and `smoke-10pct/` directories are historical
selection evidence, not release candidates.

## Reproduction

From the locked project environment:

```powershell
.\.conda\python.exe scripts\fetch_tokenizer_datasets.py --profile mvp --use-cache --offline --resume --max-memory-gib 48 --min-available-memory-gib 4
.\.conda\python.exe scripts\train_tokenizer_checkpointed.py --phase all --corpus-dir data\tokenizer\corpus\mvp --state-dir <checkpoint-dir> --vocab-size 49152 --output-dir artifacts\tokenizers\mvp-tokenizer-v0 --sample-fraction 1.0 --min-frequency 2 --num-threads 16 --batch-size 1024 --seed 20260713 --max-memory-gib 80 --min-available-memory-gib 8
.\.conda\python.exe scripts\evaluate_tokenizers.py --corpus-dir data\tokenizer\holdout\mvp --sample-dir data\tokenizer\evaluation\mvp-v0 --report-dir artifacts\tokenizers\reports\mvp-tokenizer-v0 --candidate mvp-48k=artifacts\tokenizers\mvp-tokenizer-v0 --candidate fallback-32k=artifacts\tokenizers\50pct\mvp-32k --sample-size 500 --long-quota 25 --seed 20260713 --rebuild-samples
.\.conda\python.exe scripts\freeze_tokenizer_artifact.py
```

The source lock, dataset configuration, dependency lock, fixed seed, and
training metadata are part of the provenance contract. CTranslate2 publishing
is a downstream deployment check and was intentionally not a prerequisite for
this bounded tokenizer retraining/freeze.

## Known limits

- Rare Unicode outside the must-cover alphabet can still map to `<unk>` because
  byte fallback is disabled. The freeze report measures source-character loss
  with offset mappings rather than only counting `<unk>` tokens.
- The 32k fallback has no native Traditional Chinese training data and should
  not replace `mvp-tokenizer-v0` for five-language model training.
- Translation quality and CTranslate2 performance belong to downstream model
  and deployment validation; a random micro M2M100 forward only verifies the
  tokenizer/model ID-space contract.

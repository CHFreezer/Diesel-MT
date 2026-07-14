# CTranslate2 deployment validation artifacts

`runtime/` contains rebuildable local Hugging Face checkpoints, converted
CTranslate2 models, and later offline-package staging directories. It is
Git-ignored because these binary artifacts are generated from committed code,
configuration, and the frozen tokenizer.

All machine-readable phase results are merged into
`deployment-validation.json`. Narrative evidence is kept with the completed
task documents and unified review under `work/done/`; no standalone report
Markdown is generated.

Build the TD-01 checkpoint from the repository root:

```pwsh
.\.conda\python.exe scripts\build_micro_m2m100_checkpoint.py --overwrite
```

Run TD-02 through TD-05 in their strict order and update the same consolidated
JSON:

```pwsh
.\.conda\python.exe scripts\validate_ctranslate2_deployment.py --phase all --overwrite
```

Individual phases are `convert`, `vocab`, `smoke`, and `package`. A failed run
writes `runtime/logs/last-failure.json` and does not publish an incomplete
manifest. The successful offline package is laid out as independent
`tokenizer/` and `model/` directories under `runtime/offline-package-int8/`.

The package smoke process uses Hugging Face offline flags, dead local proxy
endpoints, and a Python socket guard. It runs from a clean temporary working
directory and loads all model/tokenizer files from the deployment root.

The generated model is randomly initialized. It validates deployment
interfaces and token-ID compatibility only; it has no translation quality.

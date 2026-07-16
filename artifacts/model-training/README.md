# Diesel-MT model-training artifacts

This directory separates durable validation evidence from local runtime assets.
Task numbers belong to workflow records under `work/`; they are not storage
categories and are intentionally omitted from artifact paths.

## Reports

`reports/` contains the compact, publishable evidence for the model-training
workflow, grouped by the system or release stage it describes:

- `m0/`: human-corpus acceptance evidence for the historical M0 v1 release and
  the current 20-route M0 release.
- `teacher/`: teacher runtime selection, prompt/decode calibration, and
  distillation evidence. Individual D0, D1, Chinese-conversion, and 20-route
  composite reports remain separate because they bind different immutable
  inputs and manifests.
- `student/`: student construction, training smoke, exact-resume, M1 overfit,
  and evaluation-protocol validation.
- `m2/`: the frozen resource profile and the source-matched human/distilled A/B
  contract. Raw candidate, soak, and resume measurements are grouped under
  `m2/resources/`.

These reports are not merged into one large JSON file: each report has its own
schema and hash boundary, while the directory hierarchy provides the unified
report view.

## Runtime

`runtime/` contains large or executable local assets rather than release
reports:

- `teacher/`: verified teacher snapshots, pinned runtimes, environments, and
  runtime-local diagnostic output.
- `student-checkpoint/`: the deterministic local student checkpoint used by
  construction and encoding validation.
- `m2-resource-candidates/`: generated candidate training configurations used
  by the resource benchmark.

Runtime assets are Git-ignored and may be rebuilt or restored from their locks.
The durable conclusions derived from them live under `reports/`.

## Related data

The current 20-route model corpus is published at `data/model/corpus/mvp/`.
The immutable 18-route M0 input still required by the frozen D0/D1 evidence is
kept explicitly at `data/model/history/m0-v1/`. Temporary build state belongs
under `data/model/interim/` and must not be treated as a published artifact.

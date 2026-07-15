# task TD-01: 冻结执行契约、目录与 Git 边界

状态：in_progress（9 组/18 路 v1 已完成；20 路 amendment 进行中）

依赖：无

## 目标

建立模型训练工作流的唯一 schema、方向矩阵、配置身份、目录布局和 Git 边界，使后续数据、teacher、student、训练与部署 task 共享同一套可验证契约。

## 输入

- [MVP model training plan](../../plan/mvp-model-training.md)
- [MVP model training todo](../../todo/mvp-model-training.md)
- [`mvp-tokenizer-v0`](../../../artifacts/tokenizers/mvp-tokenizer-v0/)
- 现有 `.gitignore`、配置和测试布局

## 原子边界

本 task 只冻结接口、schema、路径和配置骨架，不下载真实语料、不加载 teacher、不训练 student，也不修改冻结 tokenizer。

## 执行事项

- 定义规范平行样本 schema，至少包含 `sample_id`、`sample_group_id`、`source_id`、`source_version`、`license`、`src_lang`、`tgt_lang`、`source_text`、`target_text`、`split`，并为 teacher/转换增强样本定义生成 provenance。
- 保留 5 标签、9 组/18 路 v1 身份；版本化扩展为 10 组/20 路，使 `zho_Hans <-> zho_Hant` 合法，同标签和 allowlist 外路由继续拒绝。
- teacher 语言名称继续使用 `Chinese` / `Traditional Chinese`，不增加 locale-specific prompt/token，不修改冻结 tokenizer。
- 固定 `data/model/raw/`、`cache/`、`interim/`、`corpus/mvp/`、`reports/` 以及 SSD 热 checkpoint/staging 与最终发布边界。
- 更新 Git ignore 边界：大体积数据和运行产物默认不跟踪，只提交 schema、配置、lock、fixture、精简报告与文档。
- 定义 `configs/mvp_model_data.yaml`、`configs/mvp_e8_d2_v48k.yaml` 的字段、schema version、稳定序列化和配置哈希规则。
- 为 schema、方向矩阵、路径边界、未知/缺失字段和非法路由增加配置级自动化测试。

## 产物

- 数据与训练配置骨架及其 schema/加载器。
- 目录、staging、发布和 Git 边界说明。
- 方向矩阵与配置验证测试。

## 验收

- 同一配置可唯一确定允许的数据形态、模型身份、路径与产物边界。
- 4 产品语言、5 标签、10 组、20 路、12 个跨语言方向与 2 个简繁互转操作的术语无歧义。
- 非法路由、未知字段、缺失字段和越界路径均明确失败。
- 冻结 tokenizer 内容及根哈希未变化。

## 实现与验收记录（2026-07-15）

- 新增严格契约模块 [`model_training_contract.py`](../../../scripts/model_training_contract.py)，统一定义 4 产品语言、5 模型标签、9 无向组、18 路由和 12 产品方向，并对非法路由、schema、provenance、路径和 lock fail-fast。
- 新增 [`mvp_model_data.yaml`](../../../configs/mvp_model_data.yaml) 与 [`mvp_e8_d2_v48k.yaml`](../../../configs/mvp_e8_d2_v48k.yaml)。解析后使用规范 UTF-8 JSON + SHA-256；当前 data config hash 为 `4b774c6d564b02fef3d6113d3de4b51428248646fe209a9e5a300c9608cb5c93`，student config hash 为 `e2def019a9eb67ab56ea2e2d3432ffaee87aa6ed36186cb551f6e7ce473732d1`。
- 数据目录与训练 artifact 边界已写入 [`.gitignore`](../../../.gitignore) 和 [`model-training-contract.md`](../../../docs/model-training-contract.md)；逐路径 `git check-ignore -v` 验证 raw/cache/interim/corpus/report、热运行和 HF/CT2 权重均被忽略。
- 默认热运行根为 `artifacts/model-training/runtime/`；正式本机运行可通过 `DIESEL_MT_MODEL_RUNTIME` 指向绝对 SSD 路径，解析路径必须进入 run manifest，不进入语义 config hash。
- 专项测试 [`test_model_training_contract.py`](../../../tests/test_model_training_contract.py) 为 `23 passed`，覆盖未知/缺失字段、路径逃逸、非法路由、三类 provenance、source lock、student 身份和 tokenizer 冻结。
- `artifact_manifest.json` SHA-256 复核仍为 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`；未修改 tokenizer 内容。

以上是不可变 v1 完成记录。2026-07-16 因新增两条简繁互转路线，本 task 退回 `in_progress`；新校验器、配置哈希、20 路 fixture 和回归完成前，不得供 TD-03/TD-09 做完整范围验收。

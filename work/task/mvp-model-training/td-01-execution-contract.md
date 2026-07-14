# task TD-01: 冻结执行契约、目录与 Git 边界

状态：pending

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
- 固定 5 个允许标签、9 组无向标签对和 18 个有向路由；明确拒绝同标签、`zho_Hans <-> zho_Hant` 和 allowlist 外路由。
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
- 5 标签、9 无向组、18 路由、12 产品方向的术语无歧义。
- 非法路由、未知字段、缺失字段和越界路径均明确失败。
- 冻结 tokenizer 内容及根哈希未变化。

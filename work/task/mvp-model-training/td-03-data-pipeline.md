# task TD-03: 实现确定性平行数据构建管线

状态：completed

依赖：TD-01、TD-02

## 目标

实现可 dry-run、可恢复、可离线重建且原子发布的平行数据构建管线，将锁定来源转换为统一 UTF-8/LF 规范样本和可追溯 manifest。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01 schema/路径契约
- TD-02 registry、source lock 与许可结论

## 原子边界

本 task 负责来源适配、获取、规范化、保守清洗和规范样本发布；分组 split、跨集合去重/泄漏防护留给 TD-04，正式 M0 运行验收留给 TD-05。

## 执行事项

- 实现薄 CLI `scripts/prepare_model_data.py` 和核心模块 `scripts/model_data_pipeline.py`，保持仓库扁平模块结构。
- 支持 dry-run、source lock 校验、下载/断点续传、缓存复用、完全离线重建和失败后安全恢复；正式构建不得解析浮动 `latest`。
- 将不同来源解析为规范样本，以稳定来源身份和规范内容生成 `sample_id`/`sample_group_id`；禁止使用 Python `hash()`、绝对路径或时间戳作为内容身份。
- 实现保守清洗：Unicode/空白规范、空文本、控制字符、HTML 残留、错误脚本占优、长度/长度比和异常内容过滤；禁止小写化、简繁转换、假名转换和韩文转写。
- 对原生、人工平行、teacher synthetic 和脚本转换增强样本保留不同 provenance。
- 输出 UTF-8/LF JSONL、拒绝原因统计、来源/标签对计数和原子 manifest；逐文件记录大小/SHA-256，manifest 最后发布。
- 用小型 fixture 覆盖全部来源适配器、缓存损坏、网络失败、恢复和半成品清理。
- 使用新的 10 组合同让每个 MASSIVE alignment group 增加 `zho_Hans--zho_Hant` human relation，更新 fixture、统计和 manifest；不得覆盖 v1 corpus。

## 产物

- `scripts/prepare_model_data.py`、`scripts/model_data_pipeline.py`。
- 来源 fixture、规范 corpus/manifest schema 和构建报告。
- 数据管线单元与失败路径测试。

## 验收

- 锁定缓存可在断网条件下重建相同规范样本。
- 同一输入身份产生稳定 sample/group ID 和规范字节。
- 清洗不改变语言脚本语义且 provenance 不丢失。
- 任何失败都不会发布可被误认为 complete 的 corpus/manifest。

## 完成记录

- 实现 [`prepare_model_data.py`](../../../scripts/prepare_model_data.py) 薄 CLI 与 [`model_data_pipeline.py`](../../../scripts/model_data_pipeline.py) 核心模块；严格消费 TD-01 config/schema 和 TD-02 source lock，不解析浮动版本。
- 锁定归档缓存支持字节范围断点续传、大小/SHA-256 双校验和完全离线复用；tar 只读取 lock 中唯一的普通文件成员，并再次校验所选文件大小/SHA-256，不把归档路径解压到文件系统。
- MASSIVE adapter 以 `(partition,id)` 对齐五个 locale，在反向扩展前按配置顺序产生 9 个无向样本；同一多平行关系共享绑定来源、alignment key 和五侧内容哈希的 `sample_group_id`，每个标签对另有内容绑定的 `sample_id`。
- 清洗 profile `td03-v1` 采用 NFC 和 Unicode 空白折叠，只拒绝内容无效项；profile SHA-256 为 `3d44d0e609d2cea22bb3d1ffb65b8de747254048208125caedb8203a529d5651`。没有小写化、简繁转换、假名转换或韩文转写。
- 规范产物为 `human_parallel.jsonl`、原始许可/NOTICE、拒绝统计、构建报告和最后发布的 `manifest.json`。manifest 记录所有文件大小/SHA-256，且 fresh、独立输出目录与 checkpoint resume 的规范文件字节一致。
- 运行说明与 corpus/manifest schema 见 [`model-data-pipeline.md`](../../../docs/model-data-pipeline.md)。小型 MASSIVE 五 locale fixture 覆盖 train/dev/test、缓存损坏、网络失败、断点续传、checkpoint 恢复、成员哈希错误和发布中断。
- 专项验证：`.conda\python.exe -m pytest tests/test_model_data_pipeline.py tests/test_model_training_contract.py -q`，结果 `33 passed in 0.75s`。
- 全量离线验证：`.conda\python.exe -m pytest -q`，结果 `85 passed in 22.92s`。
- 正式 40,251,390-byte MASSIVE 归档下载、不同 cache/worker 状态的真实规模双构建、人工抽检与 M0 发布决定仍按原子边界留给 TD-05；TD-03 的完成不代表真实语料已验收。

以上是 v1 完成记录。

## 10 组 schema v2 完成记录（2026-07-16）

- 最初在隔离构建根生成 164,778 条清洗后无向记录，拒绝 432 条；每个完整 alignment group 生成第 10 个 `zho_Hans--zho_Hant` relation。验收完成后，最终 20 路数据已发布到标准 `data/model/corpus/mvp/`，隔离构建根已清理。
- TD-03 manifest SHA-256 为 `113a33afa2ca6f73e8e10fbd5a3dab876dd470fbf0e570320edb0961901fe0c7`，构建报告 SHA-256 为 `8718f7e494580c79377f1b614b12d5a7e7ff34ae7b11a570006963341dd843c1`。
- 使用相同锁定缓存在第二个独立根完成完全离线 fresh build 和五 locale resume；所有规范产物逐字节一致，v1 corpus 未覆盖。

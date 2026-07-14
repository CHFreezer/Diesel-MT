# task TD-03: 实现确定性平行数据构建管线

状态：pending

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

## 产物

- `scripts/prepare_model_data.py`、`scripts/model_data_pipeline.py`。
- 来源 fixture、规范 corpus/manifest schema 和构建报告。
- 数据管线单元与失败路径测试。

## 验收

- 锁定缓存可在断网条件下重建相同规范样本。
- 同一输入身份产生稳定 sample/group ID 和规范字节。
- 清洗不改变语言脚本语义且 provenance 不丢失。
- 任何失败都不会发布可被误认为 complete 的 corpus/manifest。

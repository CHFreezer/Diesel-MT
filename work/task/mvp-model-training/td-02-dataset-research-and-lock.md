# task TD-02: 调研并锁定有界平行数据来源

状态：pending

依赖：TD-01

## 目标

为 9 组无向标签对确定许可清晰、版本可锁定、规模有界且可审计的人类平行语料方案，并发布后续构建唯一消费的 source lock。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01 的 schema、方向矩阵和配置骨架
- 候选语料的数据卡、许可证、版本与下载入口

## 原子边界

本 task 只完成来源研究、预算和 lock，不实现下载/清洗管线，也不把未审清许可的候选纳入正式配置。

## 执行事项

- 对 9 组无向标签对记录下载版本、语言/脚本标注、许可证、数据卡、对齐质量、规模和获取方式，优先选择许可清晰的人类平行语料。
- 对 3 组繁体相关语料确认繁体侧为原生 `zho_Hant`，不得把简转繁、`yue_Hant` 或脚本未知中文静默归类为普通话繁体。
- 对确实缺少人类语料的标签对设计有界 synthetic 补充方案，保留原生文本侧与完整 teacher provenance，且不扩张为全量蒸馏。
- 为每组冻结 train/dev/test 最小样本预算、扫描上限和下载上限；繁体预算可较低但 dev/test 不得为空。
- 生成来源 registry 与 `configs/mvp_model_data.lock.json`，锁定 URI、版本、大小、SHA-256、许可证和逻辑处理顺序。
- 列出并排除许可不兼容、用途不明或无法稳定版本化的候选。

## 产物

- `docs/model-training-dataset-research.md`。
- 数据来源 registry 与 `configs/mvp_model_data.lock.json`。
- 9 组覆盖、预算、许可和排除矩阵。

## 验收

- 9 组语料都有明确、可审计、规模有界的来源方案。
- 每个正式来源均有稳定版本、SHA-256、许可证和处理顺序。
- 原生繁体身份与 synthetic 边界单独可查。
- 任一未关闭的来源或许可缺口都会阻塞 TD-03。

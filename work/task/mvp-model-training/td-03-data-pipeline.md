# task TD-03: 实现确定性平行数据构建管线

状态：completed（ability-first source bank 与 human anchors 已在独立运行根原子发布；旧 MASSIVE corpus 未覆盖）

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

## v3 重新执行边界（2026-07-17）

- 保留现有下载、缓存、原子发布和稳定 ID 实现；为 TD-02 新锁定来源增加必要 adapter，不覆盖旧 MASSIVE 产物。
- schema/provenance 必须携带来源用途和 fidelity policy；`localization_parallel` 不得静默进入 `literal_parallel` 主体。
- 若保留 MASSIVE，审计 `utt`、`annot_utt`、slot 与 locale adaptation；只有可证明忠实或使用可逆统一 placeholder 的记录才可进入 literal MT。
- 新 corpus 以全新 identity 发布，并报告 independent semantic groups、唯一文本和 token 规模；完成后才交给 TD-04 v3。

## ability-first v4 完成记录（2026-07-17）

- 新实现为 [`mvp_60m_data_pipeline.py`](../../../scripts/mvp_60m_data_pipeline.py) 与 [`build_mvp_60m_source_bank.py`](../../../scripts/build_mvp_60m_source_bank.py)。构建根为 `D:\Diesel-MT-Runtime\mvp60-data-v1\td03`，manifest 最后发布；仓库内旧 M0/MASSIVE corpus 未修改。
- TD-03 启动前对 HPLT 四个非繁体分片执行确定性抽样，现有门仍会放进中文网页拼接、日文借贷 SEO、英文营销残片等。为避免重演旧 M0 的来源适用性错误，translation source 改用已 byte-lock 的人工平行语料单语侧：ALT 新闻、KFTT、韩英新闻与 UNPC；所有 source/anchor 按 semantic group 分区。
- source bank 共 200,851 条：`eng_Latn/zho_Hans/jpn_Jpan/kor_Hang` 各 50,000，原生 `zho_Hant` 质量实收 851；规范 JSONL SHA-256 为 `ae590f476125461f3f2acf7e3230ae0e6e72215622fb4ec6fe1419448f582e4e`。
- human anchors 实收 40,000 条、12,000 个独立组：ALT 4,000 组三语双向展开 24,000 条，KFTT 3,000 组 6,000 条，韩英新闻 3,000 组 6,000 条，MOJ 2,000 组 4,000 条。MASSIVE 的 230 个全五侧质量组均与 498 条原生繁体 source 的 semantic group 冲突，因此按“不重用、不回填”合同实收 0，而不是放宽门；anchor JSONL SHA-256 为 `82b23f97ef7e7a38ea6b1aac0f9ee3099e29d16ef0f1dfd552f6ce094383715c`。
- 两次物化专门复核 MOJ article group key；最终验证 `source_anchor_group_overlap=0`、跨集合 exact/near overlap=0、FLORES dev contamination=0、zero truncation=true，且从未读取正式 devtest。
- 紧凑机器证据为 [`mvp-60m-td03-manifest.json`](../../../artifacts/model-training/reports/m0/mvp-60m-td03-manifest.json)。TD-04 只能消费该 manifest 绑定的 source bank 和 human anchors。

## ability-first v4 英文实体修订完成记录（2026-07-17）

- 原 `mvp60-data-v1` 的 TD-03 source bank 作为被否决 TD-04 的输入保留；未覆盖其 runtime。新运行根为 `D:\Diesel-MT-Runtime\mvp60-data-v2`。
- 新 source bank 仍为 200,851 条，SHA-256 `4702ac9659483a361e0dbf663bf39f434df221ff0d8a818343f53adc0d22843a`。EN=UNPC 30k + ALT 5k + 韩英新闻 15k，KFTT EN=0；Hans=UNPC 46k + ALT 4k；JA/KO/Hant 不变。
- 该 source bank 已被 TD-04 v3 人工质量门否决为 teacher publication 输入：KFTT JA 49k 在生成英文时系统性破坏专名、年号和术语。其字节身份与运行证据继续保留，但下一 TD-03 identity 必须显式区分 KFTT human JA–EN pair 与需要 teacher 翻译的 source，不能原样复用本 manifest。
- human anchors 40,000 条完全不变，SHA-256 仍为 `82b23f97ef7e7a38ea6b1aac0f9ee3099e29d16ef0f1dfd552f6ce094383715c`。
- 独立复核确认所有标签之间 source semantic-group overlap=0、UNPC 英简 alignment-line overlap=0、source/anchor group overlap=0、exact/near overlap=0、FLORES dev contamination=0、zero truncation=true；从未读取 devtest。
- runtime manifest SHA-256 为 `7795f2bb5d8c18cff78a3cbb751f913dd6d1a1470923b656667d7a6def1ec8c1`；提交内 compact manifest SHA-256 为 `c7bff0a6bdc811b0b06358c84ac5accad3922b3746239fa70292810fc7ed4bc4`。

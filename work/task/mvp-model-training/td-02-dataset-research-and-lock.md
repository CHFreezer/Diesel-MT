# task TD-02: 调研并锁定有界平行数据来源

状态：in_progress（历史 v1/v2 已完成；通用 MT v3 来源调研已重新打开）

依赖：TD-01

## 目标

为 10 组无向模型关系确定许可清晰、版本可锁定、规模有界且可审计的人类平行/本地化语料方案，并发布后续构建唯一消费的新 source lock。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01 的 schema、方向矩阵和配置骨架
- 候选语料的数据卡、许可证、版本与下载入口

## 原子边界

本 task 只完成来源研究、预算和 lock，不实现下载/清洗管线，也不把未审清许可的候选纳入正式配置。

## 执行事项

- 保留 9 组 v1 调研；把锁定 MASSIVE `zh-CN`/`zh-TW` 文件登记为 `zho_Hans--zho_Hant` 第 10 组，无需重新下载。
- 对 3 组繁体相关语料确认繁体侧为原生 `zho_Hant`，不得把简转繁、`yue_Hant` 或脚本未知中文静默归类为普通话繁体。
- 对确实缺少人类语料的标签对设计有界 synthetic 补充方案，保留原生文本侧与完整 teacher provenance，且不扩张为全量蒸馏。
- 为每组冻结 train/dev/test 最小样本预算、扫描上限和下载上限；繁体预算可较低但 dev/test 不得为空。
- 生成来源 registry 与 `configs/mvp_model_data.lock.json`，锁定 URI、版本、大小、SHA-256、许可证和逻辑处理顺序。
- 列出并排除许可不兼容、用途不明或无法稳定版本化的候选。

## 产物

- `docs/model-training-dataset-research.md`。
- 数据来源 registry 与 `configs/mvp_model_data.lock.json`。
- 10 组覆盖、预算、许可和排除矩阵。

## 验收

- 10 组关系都有明确、可审计、规模有界的来源方案，并由新 config hash/source lock 覆盖。
- 每个正式来源均有稳定版本、SHA-256、许可证和处理顺序。
- 原生繁体身份与 synthetic 边界单独可查。
- 任一未关闭的来源或许可缺口都会阻塞 TD-03。

## 实现与验收记录（2026-07-15）

- 调研结论见 [`model-training-dataset-research.md`](../../../docs/model-training-dataset-research.md)：首轮只锁定官方 MASSIVE 1.1，它用 professional human localization 从同一 English SLURP seed 形成多平行数据，单一来源覆盖 9 个无向组。
- 官方 locale 映射冻结为 `en-US -> eng_Latn`、`zh-CN -> zho_Hans`、`zh-TW -> zho_Hant`、`ja-JP -> jpn_Jpan`、`ko-KR -> kor_Hang`。`zh-TW` 是独立人工本地化 locale，不是简转繁或 `yue_Hant`；后续仍需脚本合规和人工抽检。
- 官方 1.1 S3 归档实测为 40,251,390 bytes，SHA-256 `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`。归档内 CC BY 4.0 `LICENSE`、SLURP `NOTICE.md` 和五个 JSONL 的大小/SHA-256 均进入 [`mvp_model_data.lock.json`](../../../configs/mvp_model_data.lock.json)；lock 文件 SHA-256 为 `7508a710d93fbc874d314f455a854367bd2bcdb2b4c4ba0de82c9f35df7d8439`。
- 逐行解析确认每个 locale 都有 16,521 个唯一 `(partition,id)`：train 11,514、dev 2,033、test 2,974；其余四个 locale 与 `en-US` key 集合均为零差异。官方页面 summary 的 19,521 与 split 表/归档不一致，项目明确以锁定字节实测为准。
- 每个无向组最低 accepted 门槛为 train 10,000、dev 1,500、test 2,500；每 locale 扫描上限 16,521，下载上限 40,251,390 bytes，选中解压上限 52,000,000 bytes。
- 当前 9 组没有必须 synthetic 补足的缺口，因此本 task 不启用 synthetic。若 TD-03/TD-05 清洗后低于门槛，必须重新做来源决策并更新 config hash/lock；不得用 teacher output 或转换数据补 human dev/test。
- FLORES-200 因评测污染风险、OPUS 浮动聚合因来源/许可异质、HPLT 因无平行 alignment、自动简繁/未锁定 LLM 生成因 provenance 不满足而排除或延期，详见调研文档。
- `mvp_model_data.lock.json` 已通过严格 config-hash、来源顺序、文件身份、预算和对齐统计校验；完整离线测试为 `75 passed in 23.10s`。

以上是不可变 v1 完成记录。

## 第 10 组 lock amendment 完成记录（2026-07-16）

- 已把锁定归档中的 `zh-CN`/`zh-TW` 登记为 `zho_Hans--zho_Hant` 第 10 组人工 multiparallel localization；train/dev/test 原始上限为 11,514/2,033/2,974，无需重新下载。
- schema v2 source lock 保留官方归档、许可、notice 和五个成员文件的原始字节身份，并绑定新 config hash；lock 文件 SHA-256 为 `de24d2989ef21063a3c437b6c9bcf12362115c25d36a86cafa663e86b0ab8f88`。
- 10 组覆盖、预算与许可证缺口全部关闭，未使用 synthetic 补充 human dev/test。

## v3 重新打开原因与验收（2026-07-17）

TD-16B 长训证明此前“专业本地化即可作为通用平行翻译主体”的来源判断不成立。226,218 条 directed train records 实际来自 11,411 个 semantic/alignment groups；MASSIVE 的 locale adaptation 允许地点、媒体、服务和人物替换，不能满足通用 MT 的 source-target 忠实度。

本轮从 TD-02 重新开始，而不是在 TD-16 内修补数据：

- 重新调研覆盖 5 标签、10 组关系的真实平行来源，逐来源声明 `literal_parallel`、`localization_parallel` 或其他用途。
- 主体来源必须保留语义、实体、数字、否定和操作对象；MASSIVE 只能作为显式窄领域/路由控制补充，不得再承担通用 MT 主体。
- 数量预算同时按 independent semantic groups、唯一 `(language,text)`、token 和 directed routes 报告；路由展开不得充当新增语义。
- 为新来源锁定版本、文件 hash、许可证、下载/解压预算和处理顺序，并冻结独立 human dev/test 来源策略。
- 发布新的 config/source-lock 身份前，TD-03 v3 不得开始；旧 lock、M0 和训练证据保持不可变。

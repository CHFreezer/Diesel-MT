# task TD-02: 调研并锁定 60M MVP 数据来源

状态：completed（schema v4 ability-first 来源、繁体质量实收审计与 16 组 byte lock 已冻结；TD-03 已解锁）

依赖：TD-01

## 目标

为首个约 60M、必须达到翻译及格线的 MVP 锁定最短数据路线：四个非繁体 tag 的有界 source bank、质量门禁后实收的原生繁体、20 路 Hy-MT2 直接翻译、受控的一跳正反 pair 复用、20% human sampling weight、dev-only 弱路由补强及完整 byte lock。原生繁体数量是质量筛选的输出，不是必须完成的 KPI。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01 的 schema、方向矩阵和配置骨架
- 候选语料的数据卡、许可证、版本与下载入口

## 原子边界

本 task 只完成来源研究、预算和 lock，不生成 teacher target、不训练模型、不消费正式 test。历史 M0/D1/TD-16 证据不可覆盖。

## 执行事项

- 复用已冻结的 HPLT tokenizer train corpus 作为 EN/Hans/JA/KO teacher source，不读取 tokenizer holdout；四个 tag 各保留 50,000 的固定池。
- 为 Hant 单独审计原生繁体质量；不设 target/minimum、不 refill、不用低质量来源回填，不因已有文件或计划数量强行使用。
- Hant 技术语料最多占原生繁体实收数 15%，法律/政务最多 20%；generic `zh/cmn`、粤语/广东话、工具简转繁和老师生成文本均不得计为原生繁体。
- 前 16 条 source tag 非 Hant 的路线保持每路 10,000 accepted target；4 条 `Hant -> X` 使用原生繁体实收数决定，不设固定 accepted 数。
- 允许将已通过完整门禁的 `X -> Hant` pair 一跳反向为 `Hant -> X`，target 使用原始真人 source；反向记录不计为原生繁体、不得超过对应 outgoing-Hant 路线的 50%，正反记录必须共用 semantic group。
- human anchors 的 22,750 groups / 50,000 directed records 改为 ceiling，最终数量同样以逐来源质量门禁实收为准；训练保持 80/20 sampling weight，而不是用重复记录凑固定 raw count。
- 冻结 source/anchor/evaluation 去重、零截断、最小 hard filter、人工抽检和一次 dev-only 弱路由 patch。
- 锁定本地依赖与上游 archive 的大小/SHA-256/许可证；列出第一轮排除项。

## 产物

- `docs/model-training-dataset-research.md`。
- [`mvp_60m_distillation_sources.yaml`](../../../configs/mvp_60m_distillation_sources.yaml) 与对应 source lock。
- 20 路 teacher、五语 source bank、human anchor、预算、许可和排除矩阵。

## 验收

- EN/Hans/JA/KO 各有 50,000 条可构建 source；原生 Hant 报告 raw、逐 gate reject、dedup、domain 和最终 accepted 实收数，所有固定 quota/refill 均被合同拒绝。
- 前16路各有10,000 accepted teacher target；4条 `Hant -> X` 和全局 raw record 总数由质量实收决定。teacher/human 只冻结 80/20 sampling weight，不冻结250,000条 raw 总数。
- 一跳反向 pair 的语义/数字/实体/placeholder/semantic-group 门禁可验证，synthetic Hant 与原生 Hant 分账。
- 原生繁体候选来源、技术/法律占比上限、许可风险、formal-test 隔离和 dev-only 扩容边界单独可查。
- 新 config/source lock/hash/contract tests 一致；任一缺口阻塞 TD-03。

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

## schema v4 重新打开原因与验收（2026-07-17）

TD-16B 长训证明此前“专业本地化即可作为通用平行翻译主体”的来源判断不成立。226,218 条 directed train records 实际来自 11,411 个 semantic/alignment groups；MASSIVE 的 locale adaptation 允许地点、媒体、服务和人物替换，不能满足通用 MT 的 source-target 忠实度。

本轮从 TD-02 重新开始，而不是继续在 TD-16 内修补。“MVP”重新定义为约 60M 且达到预注册翻译及格线；流程跑通只算 smoke test。由此否决了 2026-07-17 早些时候拟定的百万级 human foundation schema v3，该草案从未提交，也不构成正式来源身份。

## schema v4 候选合同记录（2026-07-17，尚未关闭）

- 发布 [`mvp_60m_distillation_sources.yaml`](../../../configs/mvp_60m_distillation_sources.yaml)，冻结 config canonical SHA-256 为 `5a6e16d6fe55f5540f63dfb6a62d29c0b0245266dab7ff1c6d06a95a65e761fe`，状态为 `source-locked`。
- 对应 lock 文件 SHA-256 为 `408e8c7e67931c8c7bcefb3f28b430e62ea76acfa4e49e71730d63e922ab5634`，覆盖 16 组依赖：本地依赖 1,832,030,713 bytes，上游 archive 378,979,331 bytes，选中成员/有序拼接 428,040,881 bytes。UNPC 从首轮 anchor 移除，ALT 20191206 三语新闻加入；HKeL、MDN `zh-TW`、tldr `zh_TW`、UD Chinese-HK 均已 byte lock。
- EN/Hans/JA/KO 固定 source bank 合计 200,000；Hant 不再出现在固定 components 中，候选均采用 `quality-gated-actual`，不设 target/minimum/refill/低质回填。
- 前16条 source tag 非Hant的路线固定目标合计160,000 accepted；4条 `Hant -> X` 由原生实收与一跳反向 pair 共同组成，后者最多占每路50%且不计原生。human anchors 为22,750 groups / 50,000 records ceiling；训练为80/20 sampling weight。
- dev 只允许触发一次弱路由 patch，每个弱路由最多新增10,000 accepted 且禁止为达到增量而 refill；每路上限50,000、teacher全局上限1,000,000。FLORES `devtest` 在数据、teacher、训练和 patch 决策阶段全部禁止访问。
- MVP 及格线预先固定为完整 FLORES-200 dev 的 19,940 次直接路由生成：macro route chrF++ ≥25、每路 ≥12、至少 16/20 路 ≥20，且逐路由目标脚本合规率 ≥99%、空输出 ≤1%、source-copy ≤2%；必须全部满足。
- HPLT v2 parallel、ParaCrawl、WikiMatrix、human-only foundation、单语去噪、递归回译、pivot、多阶段 curriculum 和 instruction tuning 均不进入第一轮；只允许合同定义的一跳 accepted-pair 反向复用。
- [`model_data_source_contract.py`](../../../scripts/model_data_source_contract.py) 对预算、FLORES简繁脚本语义、Hant无quota/refill/低质回填、技术/法律ceiling、粤语独立排除、一跳反向provenance、零截断、holdout/formal-test隔离和16组byte lock fail closed；新增 [`mvp_60m_data_pipeline.py`](../../../scripts/mvp_60m_data_pipeline.py) 与审计 CLI。定向合同/管线测试为 `11 passed`。TD-03的唯一入口是本 task 冻结的 source bank + anchors，不是恢复旧M0训练。

## OPUS 扩展审计记录（2026-07-17）

- 已通过 OPUS API 枚举 EN/Hans/Hant/JA/KO 相关 pair；OPUS 是分 corpus、分版本、分许可的分发目录，不能把 `OPUS`、`OPUS-100` 或浮动 `latest` 当作统一来源。
- 已隔离下载并逐行抽检 ALT、Tatoeba、MDN Web Docs、tldr-pages、GlobalVoices、Wikimedia 和 TED2020 代表包；检查 raw/unique、空行、source copy、长度、目标脚本和固定语义样本，证据写入调研文档。
- 第一优先候选是 ALT EN/JA/Hans 三语交集：18,049 个唯一 English group、CC BY 4.0、抽检对齐干净。第二候选是 Tatoeba EN–JA 的有界日常短句子集，但必须保留 sentence id 与 attribution URL。
- Tatoeba EN–KO/JA–KO 只有 3,637/663 行，但机械门禁可保留 3,625/656，固定样本对齐干净；MDN EN–KO/JA–KO 清洗上限约 4,625/13,386，可与 tldr EN–KO 一起作为韩语技术/口语小比例补充，不替代现有韩英新闻主体。
- tldr-pages/MDN 只作技术域补充；GlobalVoices/Wikimedia 的当前 Moses 行存在跨句错配，未重对齐前不准作为 human anchors。
- OPUS 没有可准入且有规模的原生 Hant：TED2020/NeuLab 的 `zh_tw` 虽是真繁体，但 TED 现行条款明确禁止未经授权用于 AI/ML；Wikimedia 只有 9 条。generic `zh/cmn` 抽样混合简繁，不能映射到任一中文 tag。
- 用户确认中文采用“简体中文 / 繁体中文 / 粤语广东话”三分类：`zho_Hans`/`zho_Hant` 直接对齐冻结 FLORES-200 同名标签，繁体以台湾规范为主要输出基线，港澳正式书面繁体可补充且不破坏语义；粤语/广东话不论使用何种脚本都属于独立语言能力，当前五标签/20 路范围排除且绝不能映射到 `zho_Hant`。MOJ/MASSIVE 的 `zh-TW` 继续作为当前主体来源 provenance。
- QED、JParaCrawl、TED、News-Commentary、影视字幕类、评测集和大规模 web-mined 聚合均按许可、污染或质量原因排除首轮。ALT 是本轮唯一新增 OPUS human anchor；其 EN/JA/Hans 三语通过相同且唯一英文句连接，4,000 组最多展开 24,000 条双向记录。

## 2026-07-17 实际繁体审计与关闭证据

- 审计完整扫描 801,346 个候选片段；逐条执行 NFC、20～256 字符、frozen tokenizer 4～256 token、脚本、粤语、URL/HTML/template、机械重复、FLORES-200 dev 污染、跨源 exact/near duplicate 门。正式 `devtest` 未打开。
- domain cap 前有 450,918 条通过单条门；由于通用/日常来源本身只有 554 条最终可收，技术≤15%、法律/政府≤20% 后质量实收 851 条，而不是为了数量放宽门：MASSIVE 498、UD Chinese-HK 56、MDN zh-TW 121、tldr zh_TW 6、MOJ 170，selection SHA-256 为 `85d3ea37fa66ba76c1084e91c31d3627e74986969f5a63936df371c74bd9924c`。
- HKeL 有 3,873 条通过单条门，但在跨源法律 ceiling 中实收为 0；这不是下载/解析失败。HPLT v3 `zho_Hant` 的确定性随机审计直接发现贷款 SEO、无关文本拼接、古文和语域混杂且缺乏来源级 provenance，明确拒绝进入本轮。
- 机器可读证据为 [`mvp-60m-source-audit.json`](../../../artifacts/model-training/reports/m0/mvp-60m-source-audit.json)，SHA-256 为 `bac4bde22d979c0fe215f649782e81f86e8484c82384ab2eb4922d604a86752d`。因此 TD-02 完成，TD-03 可开始物化 200,000 条固定非Hant source、851 条原生Hant source与最多50,000条human anchors。

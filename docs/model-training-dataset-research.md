# MVP 模型训练数据集调研与来源锁定

状态：TD-02 schema v4 `in_progress`；候选合同已形成，正在继续审查 OPUS；旧 schema v2/M0/TD-16 证据保持不变

调研日期：2026-07-15；ability-first 重审：2026-07-17

## schema v4 结论：先训练出会翻译的 60M，不做 human-only foundation

TD-16B 的 20,000-step 训练说明旧路线的数据定义错了：226,218 条 directed records 只来自约 11,411 个独立 MASSIVE 语义组，并且 MASSIVE 是 locale localization，允许实体和 slot 值随地区变化。训练 loss 后期不降并不能证明训练器坏了，但足以否决“用 MASSIVE 反复展开 20 路就能得到通用翻译底模”。

新的 TD-02 不再堆百万级人工平行语料，也不再先训 human-only 底模。首个 60M MVP 只验证一条最短能力路线：**五语真实 source bank → Hy-MT2 对其余四个标签直接翻译 → 少量人工锚点混训 → FLORES dev 选择 → 最多一次弱路由补强**。单语去噪预训练、递归回译、pivot、多阶段 curriculum 和翻译指令微调全部推迟到 60M 配方通过以后；唯一例外是同一个已验收 pair 的一跳正反向复用。

新的唯一配置是 [`mvp_60m_distillation_sources.yaml`](../configs/mvp_60m_distillation_sources.yaml)，byte lock 是 [`mvp_60m_distillation_sources.lock.json`](../configs/mvp_60m_distillation_sources.lock.json)。它们不会覆盖旧 [`mvp_model_data.yaml`](../configs/mvp_model_data.yaml)、旧 M0、D1 或 TD-16 checkpoint。

## 首轮预算边界：繁体按质量实收

| 部分 | 规模边界 | 含义 |
| --- | ---: | --- |
| EN/Hans/JA/KO source bank | 固定 200,000 texts | 每个 tag 50,000 |
| 原生 Hant source bank | 无 target/minimum | 严格门禁后实收；不 refill、不低质回填、不用 synthetic 冒充 |
| 首轮 teacher 固定部分 | 160,000 accepted records | source tag 非 Hant 的16路各10,000；直接翻译、不经 English pivot |
| `Hant -> X` teacher | 质量实收 | 原生 Hant 直接生成 + 最多50%一跳反向 pair；不设每路固定数 |
| human anchors | 最多 50,000 directed records | 22,750 groups / 50,000 records 均为 ceiling |
| 训练混合 | 80% teacher / 20% human sampling weight | raw corpus 总数由实收决定，不复制记录凑250,000 |
| dev 弱路由补强 | 每弱路由最多 +10,000 | 最多一次 patch；禁止为了达到增量而降低门禁/refill |

EN/Hans/JA/KO 各从 50,000 source 池为四个目标路由选择候选，因此16路仍有每路10,000 accepted 的固定能力预算；硬门禁不降低，单路扫描12,000仍不足时必须阻塞。Hant 不使用该 quota：同一条高质量原生 Hant 可以在不同目标路由复用并共用 semantic group，但同一路线不得复制填数。

## source bank 选择

已经冻结的 tokenizer train corpus 可直接复用，不重新下载 HPLT。探索性二次门禁采用 30–600 字符、URL/email/HTML/spam 排除和固定 SHA 排序，确认四个桶有充足候选：English 289,203、Hans 1,064,563、Japanese 950,898、Korean 514,238，均远高于各 50,000 的需求。正式 TD-03 会进一步收紧到 20–256 字符、4–256 个冻结 tokenizer token，任何 overflow 都拒绝而不截断。

Traditional Chinese 是例外。对冻结 HPLT Hant 的固定抽样出现贷款 SEO、拼接乱码、随机汉字和古文混杂；manifest 自身也显示 exact duplicate rate 44.72%、approximate duplicate rate 7.40%，明显高于另外四桶。此前“40,000 MOJ 法条 + 10,000 MASSIVE”的固定配额被否决：它会为了数量造成法律/助手域失衡，也会把 accepted count 错当 KPI。

新政策是 **原生繁体能通过多少就收多少**：不设20k/25k/50k目标，不设最低数，不从低质量 HPLT、技术文档或法律条文回填。技术文本最多占原生繁体实收数15%，法律/政务最多20%；这些是 ceiling，不是必须用满。候选按通用 HPLT 严格重筛、MASSIVE `zh-TW`、台湾 MOJ、香港电子法例、MDN `zh-TW`、tldr `zh_TW` 与 UD Chinese-HK 分账审计；尚未 byte-lock 的来源不能提前计数。

其他四个源语言向 `zho_Hant` 生成的 teacher target 提供广域繁体目标侧表达。通过完整语义/数字/实体/placeholder 门禁后，可以把其中一部分 pair 反向为 `Hant -> X`：反向 source 是显式 synthetic Hant，target 是原始真人 source；它不计入原生繁体，最多占对应 outgoing-Hant 路线的50%，与正向记录共用 semantic group。这是一跳配对复用，不允许递归回译链。

能力定义采用“简体中文 / 繁体中文 / 粤语广东话”三分类。`zho_Hans` / `zho_Hant` 直接对齐冻结 FLORES-200 的同名标签，分别表示现代标准书面中文的简体/繁体；繁体以台湾规范作为主要输出基线。来自香港或澳门的正式书面繁体只要不是粤语、许可和脚本身份明确，也可进入后续候选，不视为破坏繁体语义。粤语/广东话无论以繁体还是简体书写都属于单独语言能力，当前模型完全排除，不能借 `zho_Hant` 混入。

HPLT 官方说明 packaging 使用 CC0，但原始网页文本不属于 HPLT，使用者仍负责遵守适用法律。因此它适合内部 MVP 研究，不等同于商业发布已经获得所有网页权利；发布模型前必须另做 provenance/attribution review。[HPLT 3.0 terms](https://hplt-project.org/datasets/v3.0)

## OPUS 二次审计：可用的是少量独立 corpus，不是 OPUS 整包

2026-07-17 按 OPUS API 的 `latest + moses` 身份枚举了 `en/zh/zh_tw/ja/ko` 相关语言对，并对 ALT、Tatoeba、MDN Web Docs、tldr-pages、GlobalVoices、Wikimedia Content Translation 和 TED2020 的代表包做隔离抽检。OPUS 当前汇总超过一千个 corpus，但每个 corpus 有独立来源、版本和许可；因此 `OPUS`、`OPUS-100`、`latest` 或第三方合并包都不能成为单一 source identity。[OPUS corpus 目录](https://opus.nlpl.eu/corpora)、[OPUS 官方仓库](https://github.com/Helsinki-NLP/OPUS)

抽检不是只看网页标称句数。对下载的 Moses 双文件逐行检查行数一致、空行、exact pair duplicate、source copy、长度比和目标脚本，再做固定样本语义检查。下表的“机械可保留”只经过这些可解释 hard gates，仍不是最终 accepted 数；GlobalVoices/Wikimedia 的语义抽样证明机械门禁会高估真实可用量。

| corpus / pair | OPUS raw | 机械可保留 | 许可与来源 | TD-02 判断 |
| --- | ---: | ---: | --- | --- |
| ALT v20191206 EN–JA | 18,083 | 18,081（100.0%） | CC BY 4.0；English Wikinews 由项目译为多语 | **A 级**；与 EN–Hans 交集形成 18,049 个唯一三语 group，可作通用新闻 human anchors |
| Tatoeba v2023-04-12 EN–JA | 216,046 | 215,469（99.7%） | CC BY 2.0 FR；社区翻译句 | **A 级但须归属账本**；保留 XML sentence id，并为每句生成 Tatoeba URL/作者归属，不只保留无元数据纯文本 |
| Tatoeba EN–KO / JA–KO | 3,637 / 663 | 3,625 / 656（99.7% / 98.9%） | 同上 | **B 级小规模韩语补充**；量小但抽样对齐干净，可补口语、习语和语域，不单独承担韩语主体 |
| tldr-pages v2025-11-24 EN–KO | 52,882 | 24,965（47.2%） | CC BY 4.0；社区维护 CLI 文档 | **B 级**；去重后作技术域 source/anchor 小比例补充，不承担通用主体 |
| MDN Web Docs v2023-09-25 EN–JA | 152,088 | 53,692（35.3%） | CC BY-SA 2.5；MDN 人工文档翻译 | **B 级**；原包约 43% duplicate/source-copy，必须去代码、模板和未翻译片段后小比例使用 |
| MDN EN–KO / JA–KO | 7,824 / 30,165 | 4,625 / 13,386（59.1% / 44.4%） | 同上 | **B 级韩语技术域补充**；真实译句可用，但 code/source-copy/模板比例高，必须与 EN–JA 共用严格门禁 |
| GlobalVoices v2018q4 EN–KO | 9,382 | 8,591（91.6%） | Global Voices 新闻翻译，原文要求归属 | **C 级待重对齐**；固定样本出现跨句错配，不能直接把 Moses 行当 human anchor |
| wikimedia v20230407 EN–JA | 291,563 | 224,387（77.0%） | CC BY-SA 4.0；Wikimedia Content Translation | **C 级 source-only reserve**；长段错配、本地化改写和标题片段明显，不作为 literal human anchor；单语侧清洗后可作 teacher source reserve |
| TED2020 v1 EN–`zh_tw` | 404,726 | 未准入 | TED/TEDx 志愿字幕 | **禁止训练**；现行 TED 条款明确禁止未经单独书面许可将内容用于 AI/ML，不能因 OPUS 可下载而使用 |

### 建议准入顺序

1. **ALT 先进入候选 lock。** EN–JA 与 EN–Hans 包按完全相同 English 句连接，实测得到 18,049 个唯一 EN/JA/Hans group；抽样语义忠实、重复近零，而且新闻域能直接缓解现有 UNPC/KFTT/MOJ 的领域偏斜。[ALT 数据页](https://opus.nlpl.eu/datasets/ALT)
2. **Tatoeba 以 EN–JA 为主，EN–KO/JA–KO 只作小补充。** EN–JA 的 216,046 行抽检质量明显优于大型 web-mined corpus，EN–KO/JA–KO 虽只有 3,637/663 行，机械门禁仍保留 99% 左右，适合补日常短句、习语和口语；但 CC BY 要求归属，OPUS XML 的 sentence id 必须随 record 保留，发布时可回链原句。`cmn–en` 的 47,378 行同时混有简体和繁体，不能静默映射到任一中文 tag。[Tatoeba 下载与许可字段](https://tatoeba.org/en/downloads)、[Tatoeba 使用条款](https://tatoeba.org/en/terms_of_use)
3. **tldr-pages 与 MDN 只补技术域。** tldr EN–KO 原包 52.71% exact pair duplicate；MDN EN–JA 原包 43.18% duplicate、43.25% source copy。它们在严格去重、模板/code dominance 过滤后仍有足够句子，但不应靠重复的 UI/代码字符串放大权重。[tldr-pages 许可](https://github.com/tldr-pages/tldr/blob/main/LICENSE.md)、[MDN 内容许可](https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/Attrib_copyright_license)
4. **GlobalVoices/Wikimedia 不直接进 human anchors。** 两者可提供新闻/百科 source reserve，但若要使用平行侧，必须基于 article/document identity 重做 alignment，并用语义相似度与人工样本验收；当前 OPUS Moses 行不足以证明逐句等价。[Wikimedia 内容许可](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en)

这个顺序不扩大首轮预算上限。若最终准入，它们应替换一部分当前窄域 human-anchor ceiling，而不是叠加新阶段；具体实收数要等候选 archive byte lock、三语 group 去重和 attribution ledger 通过后再写入 schema v4。

### OPUS 没有足够带身份的繁体平行数据，但这不再是 locale 阻塞

OPUS API 对 `en–zh_tw` 的当前结果只有 TED2020 404,726、NeuLab-TedTalks 218,034 和 wikimedia 9 个 alignment。抽检 TED2020 目标侧的 OpenCC traditional/simplified 变更证据比为 97.21，说明 `zh_tw` 标签确实主要是繁体，不是标签伪装；但 [TED 现行 Terms 6.3/6.4](https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use) 明确排除 AI/ML 训练，NeuLab 又来自同一 TED 字幕，二者都不能准入。剩余 9 条 Wikimedia 没有规模意义。

generic Chinese 也不能直接补这个洞：ALT 明确是 Simplified Chinese；Tatoeba `cmn` 抽检同时有大量简体与繁体（traditional/simplified 字符证据比 0.843）；tldr `zh` 虽以简体为主也混有繁体，而且 OPUS Moses 包不保留上游 `pages.zh`/`pages.zh_TW` 路径。结论是 **OPUS 可改善 EN/Hans/JA/KO 的领域和人工锚点；Hant 按 FLORES `zho_Hant` 和台湾主要规范准入标准书面繁体，港澳正式书面繁体可补充，粤语/广东话则作为独立语言排除**。若后续研究 tldr 的 `pages.zh_TW`，必须直接锁上游文件和路径，不能把 OPUS generic `zh` 重新解释成 `zho_Hant`。

OPUS 之外已验证 [香港电子法例开放数据](https://data.gov.hk/en-data/dataset/hk-doj-hkel-legislation-current) 同时发布 EN、`zh-Hant`、`zh-Hans` ZIP。隔离抽取“宪制性文件”子集后，EN/Hant 有59份同编号文档，按 `docNumber + element type + temporalId` 可得4,519个结构对齐单元，其中3,613对双方均不超过512字符；因此它可作为少量 EN/Hant、Hans/Hant 人工锚点。但它仍是法律域，只是证明来源可对齐，并不产生必须选满的配额。DATA.GOV.HK 使用条款允许商业和非商业使用并要求来源/权利归属；正式准入仍需固定下载字节与 attribution 记录。[DATA.GOV.HK 条款](https://data.gov.hk/en/terms-and-conditions)

直接上游也确认 [MDN `files/zh-tw`](https://github.com/mdn/translated-content/tree/main/files/zh-tw) 与 [tldr `pages.zh_TW`](https://github.com/tldr-pages/tldr/tree/main/pages.zh_TW) 保留明确 locale；前者当前约1,023个 Markdown 文件，后者约577个文件。它们只作为技术域候选并共享15% ceiling，不能因为路径干净就挤占通用繁体。小规模 [UD Chinese-HK](https://github.com/UniversalDependencies/UD_Chinese-HK) 明确把标准繁体中文与 Cantonese-HK 分开，可补少量日常/口语，但同样只按逐条质量实收。

### 明确排除或仅保留评测

- QED v2.0a 明示 `RESEARCH purpose only`，JParaCrawl 条款同样只许研究且限制基于数据的衍生物和商用翻译器，均不进入产品模型训练。[QED 数据页](https://opus.nlpl.eu/datasets/QED)、[JParaCrawl 官方条款](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)
- TED2020、NeuLab-TedTalks 依据现行 TED AI 条款排除；News-Commentary 的 Project Syndicate 内容受版权和商业授权控制，也不因 WMT/OPUS 再分发而获得开放训练权。[Project Syndicate 条款](https://www2.project-syndicate.org/pages/terms-and-conditions)
- JESC、OpenSubtitles 是抓取的影视字幕，数据页没有提供底层影视/字幕权利人的开放训练授权；不进入产品训练。
- TICO-19 虽为 CC0，但它是 3,071 句、多语言 COVID 翻译 benchmark；WMT-News 也是测试集。两者只可加入 contamination registry 或补充评测，不进入 train。[TICO-19 官方页](https://tico-19.github.io/)
- CCMatrix、NLLB、CCAligned、ParaCrawl、MultiParaCrawl、WikiMatrix、XLEnt 和 MultiCCAligned 是大规模 web-mined/派生集合；第一轮不为追求 raw 数量引入重复来源、错配和权利追踪成本。WikiTitles/LinguaTools-WikiTitles 主要是标题/实体，不承担句级翻译主体。

本次结论只更新 research evidence，**尚未把任何新增 OPUS corpus 写入 source config/byte lock，TD-02 保持 `in_progress`，TD-03 继续阻塞**。

## human anchor ceiling（最终以质量实收）

| 来源 | group ceiling | 展开路线 | record ceiling | 作用 |
| --- | ---: | ---: | ---: | --- |
| UNPC EN–Hans | 10,000 | 2 | 20,000 | 高忠实中英锚点 |
| KFTT EN–JA | 5,000 | 2 | 10,000 | 专业日英锚点 |
| Korean news EN–KO | 5,000 | 2 | 10,000 | 新闻韩英锚点 |
| Taiwan MOJ EN–Hant | 2,500 | 2 | 5,000 | 原生繁体英译锚点 |
| MASSIVE 五语 | 250 | 20 | 5,000 | 标签/路由控制，不当作 literal MT 主体 |

UNPC 官方将 v1.0 描述为 1990–2014 年联合国公开领域文件的人工翻译和平行句，并要求标示 UN 来源；KFTT 的 original train 为 440k 日英句，采用 CC BY-SA 3.0；台湾政府开放资料授权允许重制、改作和产品/服务利用但要求显名；韩英新闻固定 commit 明示 CC BY-SA 3.0；MASSIVE 官方说明它由 SLURP 本地化而来，因此这里只承担小比例 route control。[UNPC 官方页](https://www.un.org/dgacm/en/content/uncorpus)、[KFTT 官方页](https://www.phontron.com/kftt/)、[台湾开放资料授权条款](https://data.gov.tw/license)、[Korean Parallel Corpora](https://github.com/jungyeul/korean-parallel-corpora)、[MASSIVE 官方仓库](https://github.com/alexa/massive)

## teacher 成本与停止条件

旧 D1 的 44,480 个 raw records 累计 generation latency 为 20,385.82 秒，569,607 completion tokens，实测约 2.18 samples/s、27.94 completion tokens/s，短句 200,000 条的线性下限约 25.5 小时。新 source 平均更长，按输出 25–50 token 估算首轮应预留约 2–5 天。正因为 teacher 成本不可忽略，首轮不直接生成 1,000,000 条。

模型是否继续扩数据只看冻结 FLORES-200 `dev` 的 20 路结果，不看 train loss 是否“好看”。及格线固定为完整五标签 dev 的 19,940 次直接路由生成：SacreBLEU chrF++（char order 6 / word order 2 / beta 2）macro route ≥25、每路 ≥12、至少 16/20 路 ≥20，逐路由目标脚本合规率 ≥99%、空输出率 ≤1%、source-copy ≤2%；这些门槛必须全部通过。初次 60M 未过线时，只给弱路由增加 10,000 个 accepted target，再训练一次；若仍失败，就回到数据/模型配方审查，而不是无限生成。FLORES `devtest` 在 TD-02、teacher 生成、训练和 patch 决策中全部禁止访问；FLORES 官方将数据分成 dev、devtest 和 hidden test，本项目继续把 devtest 保留为一次性正式测试。[FLORES-200 官方 README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)

## 最小质量门禁

自动 hard reject 只保留会直接破坏监督的错误：空输出、prompt/explanation 回显、原文照抄、目标脚本错误、长度截断、异常重复和 placeholder 丢失。数字/实体漂移、异常长度比和域偏差先作为 diagnostic flag，不用脆弱启发式大规模误杀。每路固定人工检查 20 条 accepted 和最多 20 条 rejected；失败时阻塞发布并修正 gate，不静默删掉整条路线。

所有 source/anchor 还必须与 tokenizer holdout、FLORES dev/devtest、彼此 exact/near 去重。source bank 与 human anchors 按 normalized text 和 source group 双重隔离，避免同一个输入在同一路线同时出现相互冲突的人类 target 与 teacher target。

## 首轮明确排除

- HPLT v2 mined parallel、ParaCrawl、WikiMatrix：不再为第一轮引入额外 pair-fidelity 清洗复杂度；
- HPLT Hant 全桶直接准入：只允许严格重筛后逐条实收；
- human-only foundation、单语 denoising、递归回译、pivot synthetic、curriculum 和 instruction tuning：等待 60M 最短配方结论；合同定义的一跳 accepted-pair 反向复用除外；
- 一次性生成全部 1M teacher records：只有 dev 弱路由触发才增量生成；
- formal test：绝不用于数据选择、teacher 校准或训练决策。

## 历史 schema v2 记录（不可变）

## 结论

首轮有界人类平行数据只锁定官方 **MASSIVE 1.1**。它是由专业译者将 English SLURP seed 本地化到多种 locale 的多平行语料；官方论文将其描述为跨 51 种语言的平行、标注虚拟助手 utterance，并说明 50 个非英语版本由专业译者本地化。[ACL 论文与摘要](https://aclanthology.org/2023.acl-long.235/)

官方 1.1 发布同时包含 `en-US`、`zh-CN`、`zh-TW`、`ja-JP`、`ko-KR`，因此一个 `(partition, id)` group 可以在不做脚本推断、自动简繁转换或跨来源拼接的情况下形成项目所需的 10 个无向模型关系：9 个跨语言关系和 `zh-CN--zh-TW` 中文内部本地化关系。官方仓库明确提供 1.1 S3 归档、JSONL 结构、`train/dev/test` 字段和 `utt` 原文含义。[MASSIVE 官方仓库的数据说明](https://github.com/alexa/massive#accessing-and-processing-the-data)

数据归档内的 `LICENSE` 是 CC BY 4.0；`NOTICE.md` 说明 English 数据来自同为 CC BY 4.0 的 SLURP。归档、许可、notice 和五个 locale 文件的实际字节身份已经锁定，第 10 组复用同一锁定字节且无需重新下载。[`mvp_model_data.lock.json`](../configs/mvp_model_data.lock.json) 已绑定 10 组 schema v2 配置哈希 `1c3fda336a5fae183ea48e813c442daabee5b754bfbd792bad15fabaeb2c52b7`。对外再分发数据或其改编版本时必须保留归属、许可链接和修改说明；模型许可不能替代数据许可审查。

## 为什么适合本轮 MVP

- **脚本身份明确**：官方 locale 分别是 `zh-CN` 与 `zh-TW`，映射为 `zho_Hans` 与 `zho_Hant`；`zh-TW` 不是 `yue_Hant`，也不是项目运行时对 `zh-CN` 做的简转繁。
- **人工来源明确**：非英语 locale 是对同一 English seed 的专业人工本地化，不是 teacher 生成数据。
- **对齐键稳定**：五个选中 locale 的 `(partition, id)` 集合实测完全一致。
- **许可统一**：选中归档的 data/SLURP notice 都指向 CC BY 4.0。
- **范围有界**：单一 40,251,390-byte 归档即可覆盖全部 10 组，避免在 TD-02 引入大规模抓取。

局限也很明确：数据是单轮虚拟助手领域，句子较短；localization 允许按 locale 调整实体或 slot 值；日/韩/中文之间是通过同一 English seed 对齐，而不是每个非英语 pair 直接互译。因此它适合证明训练链路和语言控制，不足以支持生产翻译质量结论。

## 实测身份与结构

官方归档：`https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz`

| 项目 | 实测值 |
| --- | --- |
| 版本 | 1.1 |
| HTTP Content-Length | 40,251,390 bytes |
| SHA-256 | `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577` |
| ETag | `51e0da2a3ff7a016f109e1d1b4306e93-3` |
| Last-Modified | 2022-11-07T16:55:04Z |
| 选中数据+许可文件 | 51,782,238 bytes |
| 每 locale 总行数 | 16,521 |
| 每 locale partition | train 11,514 / dev 2,033 / test 2,974 |

官方 Hugging Face 数据卡的 summary 曾写每语言 19,521 条，但同一数据卡的 split 表、官方 1.1 JSONL 和本次逐行解析都给出 16,521（11,514 + 2,033 + 2,974）。本项目以锁定归档的实际字节与逐行统计为准，并在 lock 中记录该结果。[Hugging Face split 表](https://huggingface.co/datasets/AmazonScience/massive#data-splits)

五个 locale 文件均验证 locale 字段唯一正确、`(partition, id)` 唯一 16,521 个，且与 `en-US` 集合零差异。后续 TD-03 只读取 `utt`，不把带 slot 标注的 `annot_utt` 当普通翻译文本。

## 10 组覆盖矩阵

| 无向 pair | MASSIVE locale | train 原始上限 | dev 原始上限 | test 原始上限 | 来源类型 |
| --- | --- | ---: | ---: | ---: | --- |
| `eng_Latn--jpn_Jpan` | `en-US` / `ja-JP` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `eng_Latn--kor_Hang` | `en-US` / `ko-KR` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `jpn_Jpan--kor_Hang` | `ja-JP` / `ko-KR` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `eng_Latn--zho_Hans` | `en-US` / `zh-CN` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `jpn_Jpan--zho_Hans` | `ja-JP` / `zh-CN` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `kor_Hang--zho_Hans` | `ko-KR` / `zh-CN` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `eng_Latn--zho_Hant` | `en-US` / `zh-TW` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `jpn_Jpan--zho_Hant` | `ja-JP` / `zh-TW` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `kor_Hang--zho_Hant` | `ko-KR` / `zh-TW` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `zho_Hans--zho_Hant` | `zh-CN` / `zh-TW` | 11,514 | 2,033 | 2,974 | human Chinese-internal localization |

每组最低 accepted 门槛冻结为 train 10,000、dev 1,500、test 2,500；扫描上限为每 locale 16,521 行，下载上限为归档精确大小，选中解压上限为 52,000,000 bytes。若 TD-03/TD-05 清洗后任一组低于门槛，必须回到新的 source research 决策，不能静默用重复、简繁转换或 teacher output 补足 human dev/test。

## 繁体边界

`zh-TW -> zho_Hant` 的依据是官方 locale 和人工本地化流程，不是字符级自动分类。模型与 teacher 语言名称仍为 `Traditional Chinese`；`zh-TW` 只记录当前人类来源及其用词偏向。即便简繁共享大量字符，TD-03/TD-05 仍须做脚本、语义保持与人工抽检；locale 证据不能替代内容质量检查。

- 禁止将 `zh-CN` 自动转换后标记为原生 `zho_Hant`。
- 禁止将 FLORES 的 `yue_Hant` 或任何粤语繁体数据映射为普通话 `zho_Hant`。
- 禁止把只有 `zh` 标签、没有来源地区/脚本证据的数据静默放入任一中文桶。
- teacher synthetic 只能进入后续显式 provenance 链，不能替代本数据集的人类 dev/test。

## 未选候选与原因

| 候选 | 结论 | 原因 |
| --- | --- | --- |
| Hugging Face `AmazonScience/massive` 浮动 `main` | 不作为 source identity | 页面和 parquet 转换可继续更新，且数据卡注明上传集成人员不是原 corpus 作者；只用它交叉核对结构，正式 lock 使用官方 S3 1.1 归档。 |
| FLORES-200 | 保留为未来独立评测研究，不进入本轮 train | 官方定位是 MT evaluation benchmark，只有 dev/devtest/hidden test；将其混入 train 会污染后续标准评测。[官方 FLORES-200 README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) |
| OPUS 聚合/`latest` 查询 | 本轮排除 | OPUS 是大量不同来源和许可证的集合，必须逐 corpus/版本审查；浮动聚合不能作为一个统一许可 source lock。[OPUS corpus 目录](https://opus.nlpl.eu/corpora) |
| HPLT 3.0 monolingual corpus | 不作为人类平行来源 | 当前冻结 HPLT 数据没有跨语言 alignment key，只能服务 tokenizer/未来单语增强。 |
| 自动简繁转换或未锁定 LLM 生成 | 不作为 human source | 不能证明原生繁体，也不满足本 task 的人工平行与完整 provenance 要求。 |

当前 10 组都由同一锁定 MASSIVE 归档提供，不需要 synthetic 才能关闭来源缺口。第 10 组已经进入新 config hash/source lock 与覆盖统计；归档 URI、成员大小和 SHA-256 均未改变。后续若清洗使 accepted 数低于门槛，应新立来源变更；不得临时改义。

## TD-05 独立评测污染引用结论

TD-05 最终选择 **原版 FLORES-200** 的 `dev`/`devtest` 作为外部污染阻断引用，不选择持续更新的 FLORES+。这里的“使用”只表示在构建 M0 时检查训练候选是否与评测文本精确或近重复，绝不表示把 FLORES 文本并入 MASSIVE、训练 split、方向采样或 teacher 输入。

选择原版的理由是身份更适合当前可复现门禁：Meta 官方仓库已经归档，固定 revision `a6c830c6e1051fb4ac1a44b32358f00463f332bd` 明确列出项目五个标签并指向固定的 2022 数据归档；FLORES+ 则是仍在维护、按版本扩展的后继集合，适合后续另立评测升级决策，但不适合作为本次 M0 中途漂移的引用。[原版官方 README](https://github.com/facebookresearch/flores/blob/a6c830c6e1051fb4ac1a44b32358f00463f332bd/flores200/README.md)；[FLORES+ 数据卡](https://huggingface.co/datasets/openlanguagedata/flores_plus)

冻结身份位于 [`mvp_mt_evaluation.lock.json`](../configs/mvp_mt_evaluation.lock.json)：

- 官方归档 `flores200_dataset.tar.gz`：25,585,843 bytes，SHA-256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`；
- `eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang` 的 `dev` 997 行与 `devtest` 1,012 行，共 10,045 个单语引用记录；
- 仓库 README、benchmark README 和 `LICENSE_CC-BY-SA` 均锁定大小/SHA-256；
- `reference-manifest.json` 明确标记 `prohibited from training`，污染 registry 对它使用 `policy=block` 与 `match=exact-and-near`。

M0 正式扫描结果为 FLORES-200 `hits=0`。未来 TD-13 若真正用该集合计算模型质量，必须继续读取同一锁定身份；若改用 FLORES+，应作为显式评测版本升级，不能静默替换本锁。

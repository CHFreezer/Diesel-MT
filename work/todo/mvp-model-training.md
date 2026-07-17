# todo: MVP model training

状态：todo / active

## 来源

- plan：[MVP model training](../plan/mvp-model-training.md)
- task 索引：[MVP model training tasks](../task/mvp-model-training/index.md)
- 中文脚本/locale 能力合同：[20 路范围修正](../../docs/chinese-locale-capability-contract.md)
- 项目语言与方向口径：[README](../../README.md#语言与方向口径)
- 冻结 tokenizer：[mvp-tokenizer-v0](../../artifacts/tokenizers/mvp-tokenizer-v0/)
- tokenizer review：[mvp tokenizer review](../done/review/mvp-tokenizer.md)
- 部署兼容性 review：[CTranslate2 deployment review](../done/review/ctranslate2-deployment.md)

## 目标

使用不可变的 `mvp-tokenizer-v0`、锁定的 Hy-MT2 7B teacher 和从零初始化的 `mvp_e8_d2_v48k`，建立“五语 source bank -> 20 路直接 sequence distillation -> 80/20 teacher/human mixed 训练 -> 60M 翻译及格线 -> 重复能力等价 -> 独立评测 -> CTranslate2 float32/INT8”的第一个真实模型闭环。

本 todo 包含 source tag 非 Hant 的16路各10,000 accepted teacher targets、质量实收的4条 `Hant -> X`、一跳 accepted-pair 反向复用和最多一次 dev 弱路由 patch，但不包含无条件1M全量生成、递归回译、在线 logits/hidden-state 蒸馏、`e12-d3`、约200M模型、生产质量门槛或移动端性能优化。

2026-07-17 长训修正：高吞吐训练器已经合并，完整旧 M0 也已完成至 15k 的诊断长训；结果证明阻塞点是旧TD-02的来源适用性假设，而非 GPU/CPU 管线。新TD-02 schema v4已完成繁体质量优先来源审计和16组byte lock；TD-03已开始，旧数据、checkpoint和test隔离证据保持不变。

## 固定口径

- 产品语言为中文、英文、日文、韩文 4 种；模型标签为 `eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang` 5 个。
- 模型与 teacher 名称继续使用 `zho_Hans -> Chinese`、`zho_Hant -> Traditional Chinese`，本次扩展不增加 locale-specific 名称或控制标签。
- `zho_Hans`/`zho_Hant` 直接对齐冻结 FLORES-200 同名标签；繁体以台湾规范为主要输出基线，港澳正式书面繁体可补充。粤语/广东话不论使用繁体或简体都属于独立语言，当前五标签/20 路完全排除，绝不能映射为 `zho_Hant`。
- MVP 数据必须覆盖 10 组无向模型关系和 20 个有向训练路由：18 路跨语言翻译，以及 `zho_Hans <-> zho_Hant` 两路中文内部转换。
- 产品层仍为 12 个跨语言翻译方向，另有 2 个简繁中文互转操作。繁体必须有原生 `zho_Hant` 样本和独立 dev/test；当前来源 locale `zh-TW` 只作为 provenance，工具转换数据不能替代原生验收。
- student 只使用 49,152 词表的 `mvp_e8_d2_v48k`，从零初始化，不加载任何第三方模型权重或随机部署验证 checkpoint。
- teacher 固定为 `configs/hymt2_teacher_selection.yaml` 中的官方 Hy-MT2 7B GGUF Q8_0 + llama.cpp CUDA，只生成离散 UTF-8 译文；teacher 不进入 student 训练图，student 不继承其 tokenizer、权重或架构。
- 正式蒸馏 corpus 只从 train source 生成；仅允许在冻结的有界 human dev 子集上运行 teacher 以校准 prompt/decode，校准输出不得进入 student train，test 不得送入 teacher。
- test 只在最终候选冻结后执行一次正式评测；训练配置和 checkpoint 选择只使用 train/dev。
- 成熟度与版本号分开：M0/D1 v1 是不可变的18路历史 route/system-validation 数据，D0 v1 属于 smoke；它们都不能单独代表完整20路或通用 MT `mvp` 语料能力。
- 旧 D1 composite 的 44,361 条只进入已完成的 TD-15/原 TD-16 A/B 诊断，不自动继承到 schema v4。新 60M corpus 必须使用每路 10,000 accepted 的新 teacher identity。

## 依赖关系

```text
TD-01
├─ ability-first 数据链：TD-02 -> TD-03 source/anchors -> TD-04 teacher -> TD-05 mixed corpus
├─ teacher 运行时：TD-06
└─ student 基础链：TD-09 -> TD-10 -> TD-11

TD-05 + TD-06 -> TD-07 -> TD-08
TD-05 + TD-11 -> TD-12
TD-05 + TD-09 -> TD-13
TD-05 + TD-12 -> TD-14
TD-05 + TD-08 + TD-13 -> TD-15
TD-08 + TD-12 + TD-13 + TD-14 + TD-15 -> TD-16 A/B 诊断（已完成）
TD-16 A/B -> TD-16A -> TD-16B
TD-16B -- 旧 M0 来源适用性失败 --> TD-02 schema v4 -> TD-03 -> TD-04 -> TD-05
TD-05 -> TD-16C mixed 60M -> TD-16D 可选弱路由 patch -> TD-16E -> TD-16F
TD-16F -> TD-17 -> TD-18
```

历史执行已到 TD-16B：A/B诊断、可配置高吞吐训练器和完整旧M0长训均有证据，但TD-16B否决了旧M0作为通用MT foundation。TD-02/TD-03 schema v4 已完成来源实收、byte lock、source bank 与 human anchors；TD-04 v1 否决 KFTT 英文实体臆译，v2 因长目标截断停止，v3 则完整生成 195,404 条并通过数量门，但固定人工队列在 `jpn_Jpan→eng_Latn` 发现 KFTT 日文 source 的系统性专名/年号/术语臆译。v3 只保留为完整运行与失败诊断证据，禁止进入 TD-05；旧TD-16继续 suspended。

## 当前回退门禁

- [x] **TD-02 schema v4**：EN/Hans/JA/KO各50,000 source；原生Hant无target/minimum/refill，完整审计后实收851条；锁定一跳反向pair、human-anchor ceiling、80/20 sampling weight和一次dev-only patch。
- [x] **TD-03 schema v4**：已发布200,000条固定非Hant source、851条质量实收Hant和40,000条human anchors；严格零截断、semantic-group分区、exact/near去重和FLORES-dev隔离。
- [ ] **[TD-04 schema v4](../task/mvp-model-training/td-04-ability-first-teacher-generation.md)**：v3 已完成16路固定target、4路Hant质量实收和反向pair，但人工审查命中 KFTT 日文到英文的系统性实体/术语 blocker；当前身份 rejected，等待新的 KFTT 日英 human-pair/替代 source 合同。
- [ ] **[TD-05 schema v4](../task/mvp-model-training/td-05-ability-first-mixed-corpus.md)**：实现已完成，但 v3 人工质量门失败，publication blocked；不得消费 v3 accepted teacher 或创建80/20 mixed corpus。
- [x] 历史 TD-02～TD-05 v1/v2 产物保持不可变，只作为 route/system-validation 与失败诊断证据。

## 待办

### TD-01 冻结执行契约、目录与 Git 边界

依赖：无。

- [x] 定义规范平行样本 schema，至少包含 `sample_id`、`sample_group_id`、`source_id`、`source_version`、`license`、`src_lang`、`tgt_lang`、`source_text`、`target_text`、`split`；teacher/转换增强样本增加生成 provenance。
- [x] 冻结并完成 5 标签、9 组/18 路 v1 合同与非法路由测试，保留其历史 artifact 身份。
- [x] 固定模型数据目录：`data/model/raw/`、`cache/`、`interim/`、`corpus/mvp/`、`reports/`；所有大体积数据默认 Git-ignored，只提交 schema、配置、lock、fixture 和精简报告。
- [x] 固定训练运行目录和发布边界：热 checkpoint/staging 根可配置到高吞吐存储，完整校验后发布 Git-ignored HF/CT2 产物；提交内只保存配置、manifest、指标和文档。
- [x] 定义 `configs/mvp_model_data.yaml`、`configs/mvp_e8_d2_v48k.yaml` 的字段、schema version、稳定序列化和配置哈希规则。
- [x] 为 schema、方向矩阵、路径边界、未知字段、缺失字段和非法路由增加配置级自动化测试。
- [x] 版本化扩展 allowlist 为 10 组/20 路，使 `zho_Hans <-> zho_Hant` 两路合法，同标签和 allowlist 外路由继续 fail-fast；不得修改 v1 runtime manifest。
- [x] 保持 teacher 名称 `Chinese` / `Traditional Chinese` 和既有 tokenizer token ID，不新增 locale-specific 标签；更新 config schema、规范哈希和 20 路 fixture/反例测试。

产物：数据/训练配置骨架、目录与 Git 边界、schema/方向矩阵测试。

完成条件：新版本配置可以唯一确定 10 组/20 路数据形态、模型身份和产物位置；v1 身份与冻结 tokenizer 均未被改写。

### TD-02 调研并锁定有界平行数据来源

依赖：TD-01。

- [x] 完成 9 组 v1 来源调研和锁定。
- [x] 对繁体相关的 3 组语料逐一确认繁体侧为原生 `zho_Hant`，不把简转繁、粤语 `yue_Hant` 或脚本未知的中文静默归类为普通话繁体。
- [x] 若某组缺少足够的人类平行语料，设计有界 synthetic 补充方案：必须保留原生文本侧、teacher 身份、prompt、解码参数和生成 manifest；不得因此引入大规模蒸馏范围。
- [x] 为每组确定 MVP 的 train/dev/test 最小样本预算、扫描上限和下载上限；繁体可低于简体，但 dev/test 不得为空。
- [x] 生成来源 registry 和 `configs/mvp_model_data.lock.json`，锁定 URI、版本、文件大小、SHA-256、许可证和逻辑处理顺序。
- [x] 记录许可证不兼容、用途不明或无法稳定版本化的候选并排除，不以“能下载”代替可用性结论。
- [x] 将现有 MASSIVE `zh-CN`/`zh-TW` 对齐文件登记为 `zho_Hans--zho_Hant` 第 10 组；确认 train/dev/test 原始上限为 11,514/2,033/2,974，无需新下载。
- [x] 发布绑定新 config hash 的 source lock/覆盖矩阵，保留相同归档与成员字节身份，并记录第 10 组是人工 multiparallel localization 而非工具转换。

产物：更新后的数据 registry、source lock 和 10 组覆盖矩阵。

完成条件：10 组关系均有明确、可审计的 MVP 来源与锁定身份；任何未关闭的 config/lock 或许可缺口都会阻塞 TD-03 正式构建。

### TD-03 实现确定性平行数据构建管线

依赖：TD-01、TD-02。

- [x] 实现 `scripts/prepare_model_data.py` CLI 和独立的 `scripts/model_data_pipeline.py` 核心模块，保持仓库现有扁平模块结构。
- [x] 支持 dry-run、source lock 校验、下载/断点续传、缓存复用、离线重建和失败后安全恢复；不得在正式构建时隐式解析 `latest`。
- [x] 将不同来源解析成规范样本，使用稳定来源身份和规范内容生成 `sample_id`/`sample_group_id`，禁止 Python 内置 `hash()`、绝对路径和时间戳进入内容身份。
- [x] 实现保守清洗：Unicode/空白规范、空文本、控制字符、HTML 残留、错误脚本占优、长度/长度比和异常内容过滤；禁止小写化、简繁转换、假名转换或韩文转写。
- [x] 对原生、人工平行、teacher synthetic 和脚本转换增强数据使用不同 provenance，不允许清洗过程丢失来源类型。
- [x] 输出规范 UTF-8/LF JSONL、拒绝原因统计、来源/标签对计数和原子 manifest；manifest 最后发布且逐文件记录大小/SHA-256。
- [x] 用小型 fixture 覆盖所有来源适配器、错误路径、缓存损坏、网络失败和半成品清理。
- [x] 在新身份下让每个 MASSIVE alignment group 生成第 10 个 `zho_Hans--zho_Hant` human relation，更新 pair/潜在 route 统计与 fixture；原 v1 corpus 保持不变。
- [x] 从已校验缓存完成新版本离线构建与 resume 复验，发布独立 addendum 或 10 组 canonical corpus/manifest，不覆盖 v1 路径身份。

产物：模型数据构建 CLI/核心模块、fixture、manifest 和自动化测试。

完成条件：从已锁定缓存可完全离线重建 10 组规范样本；失败不会发布可被误认为完成的 corpus，v1 仍可按原哈希审计。

### TD-04 实现分组切分、去重与泄漏防护

依赖：TD-03。

- [x] 在扩展正反方向前按无向平行关系生成稳定 group；同一对齐关系、反向样本、同文档片段和已知派生样本必须进入同一个 split。
- [x] 在规范文本、source、target 和 pair 层执行 exact 去重；对 train/dev/test 执行跨集合 near-duplicate/污染检查并记录参数与命中原因。
- [x] split 使用稳定 group hash 和版本化比例生成，禁止逐行随机拆分；test 身份在数据构建阶段冻结。
- [x] 与 tokenizer corpus/holdout、正式 MT 评测集和同一数据源重复版本进行可追溯污染检查，不把 tokenizer holdout 当成模型质量 test。
- [x] v1 在 split 后扩展并验证 18 路反向隔离。
- [x] 验证 worker 数、缓存命中、输入完成顺序和 fresh/resume 路径不会改变 corpus、split 或 manifest 字节。
- [x] 增加反向泄漏、跨 split 近重复、派生样本、错误 group 和非确定性顺序的失败测试。
- [x] 将第 10 组绑定到既有 alignment group/component，在 split 后扩展为两条新增路线，验证 20 路反向、exact/near 和污染隔离。
- [x] 以新的固定路由顺序发布 20 路 addendum/composite manifest，并证明 worker/cache/fresh/resume 不改变字节身份；v1 finalized manifest 不变。

产物：确定性 split/dedup/leakage 模块、污染报告和自动化测试。

完成条件：20 路 train/dev/test 在 group 层相互隔离；两次独立构建的规范 corpus 与确定性 manifest 字节级一致。

### TD-05 构建并验收 M0 数据集

依赖：TD-04。

- [x] 建立并验收 9 组/18 路 v1 fixture 与 M0 v1 corpus，保留 203,942 条 human train 和既有证据。
- [x] 固定方向采样策略，报告原始样本数、过滤后样本数、正反扩展数、训练权重和有效曝光；禁止低资源方向无界重复。
- [x] 报告每个标签对/split 的来源占比、原生/synthetic/增强占比、长度与长度比分布、脚本合规率和过滤原因。
- [x] 对每组执行分层人工抽检：至少检查 20 条 accepted train、10 条 accepted dev/test 和 20 条 rejected（不足时检查全部），覆盖长度边界、繁体与混合脚本样本，并冻结精简审查记录。
- [x] 使用不同 worker/cache 状态完成两次独立构建，逐文件比较 corpus、manifest 和确定性报告 SHA-256。
- [x] 生成 M0 验收报告；任何标签对为空、繁体 dev/test 缺失、泄漏、来源不明或复现失败都阻塞训练数据发布。
- [x] 扩展 fixture 到 10 组/20 路，并为两条简繁互转路线增加合法/非法、共享汉字、词汇差异和 split 泄漏反例。
- [x] 正式构建并验收第 10 组，逐 split 报告数量、脚本、长度、语义错位和 provenance；按同一抽检预算审查 accepted/rejected。
- [x] 完成不同 cache/worker 的真实规模双构建并发布 human addendum + 20 路 composite manifest；完整 human composite 供 TD-09/TD-16B 使用，teacher 20 路 composite 只供 TD-15 与 TD-16 的 teacher-target/A/B 部分使用。

产物：不可变 M0 v1、中文内部 human addendum、20 路 fixture/composite、质量/覆盖/复现报告和完成 manifest。

完成条件：plan 的 10 组/20 路 M0 数据门槛全部满足；v1 与 addendum 身份可分别审计，只有 composite 被标记为可供完整训练链消费。

### TD-06 锁定并验证 Hy-MT2 7B teacher 运行时

依赖：TD-01。

- [x] 锁定腾讯官方 [`tencent/Hy-MT2-7B`](https://huggingface.co/tencent/Hy-MT2-7B) 或经验证的官方同模型运行 artifact，记录 Hugging Face revision、模型/代码/chat template/许可证文件清单、大小和 SHA-256。
- [x] 记录官方 [Apache-2.0 许可证](https://huggingface.co/tencent/Hy-MT2-7B/blob/main/LICENSE.txt)，并明确模型许可证不自动解决输入语料或生成数据的权利边界。
- [x] 审查并锁定官方示例要求的 `trust_remote_code` 内容；正式生成只从本地固定快照加载，启用离线标志和网络阻断，不执行浮动 `main` 或运行时下载。
- [x] 为 teacher 建立与 student 依赖隔离或明确兼容的运行 profile，锁定 Python、Transformers、PyTorch、CUDA/后端和启动命令，不让 teacher 依赖改写 student 主环境。
- [x] 在当前执行主机的 accelerator/CPU 上比较可行的官方 BF16 offload、FP8 或 GGUF 等运行路径；只选择官方来源且通过参考集验证的 artifact，不使用来源不明的社区量化。
- [x] 对 5 个项目标签完成最小离线推理，验证官方支持的 Chinese、Traditional Chinese、English、Japanese、Korean 均能生成非空结果。
- [x] 记录加载峰值内存/显存、单样本延迟、吞吐、输出稳定性和已知限制；若无可接受运行路径，D0 阻塞，不降级为其他 teacher。

产物：teacher artifact lock、remote-code/后端审查记录、离线运行 profile、五标签冒烟报告、运行时对比和冻结选型配置。

完成条件：固定 teacher artifact 可在完全离线环境重载并完成五标签推理，所有执行代码和文件身份可审计。

完成记录：TD-06 于 2026-07-15 完成。官方原版未量化 BF16、bitsandbytes 0.49.2 LLM.int8 和官方 GGUF Q8_0 + llama.cpp CUDA 均按同一 v2 协议完成可审计测评；平均吞吐分别为 4.17 / 8.79 / 27.71 tokens/s，峰值显存增量为 14,543 / 9,687 / 7,909 MiB。原版 BF16 是唯一诊断质量基线，INT8 与 GGUF 的五标签短探针均 10/10 逐字匹配；GGUF 容量探针的一处措辞差异留给 TD-07 在人类 reference 上判断。

最终冻结官方 `tencent/Hy-MT2-7B-GGUF` Q8_0 为 sequence-level 蒸馏源：revision `ab8472660ac61fac25f1af43fac2599d52a8a775`、`HY-MT2-7B-Q8_0.gguf`、SHA-256 `58b3ad55dd6f6fa08c695cddc34fb5f8f708a844f78ae10508071914b0ed67c0`、llama.cpp `b10012` CUDA 13.3。唯一规范入口为 `configs/hymt2_teacher_selection.yaml`；TD-07 负责 prompt/decode、逐路由人类 reference 质量和相对原版 BF16 的量化差异校准，失败时阻塞 D0 而不是静默更换后端。

本地存储记录：选定 GGUF/llama.cpp 与原版 BF16 基线已迁入 Git-ignored 的 `artifacts/model-training/runtime/`，迁移后分别完成实际加载冒烟；旧 runtime、FP8 权重、重复缓存、下载压缩包和临时日志已清理。该目录只允许模型顺序加载与低频只读访问，物理盘映射统一记录在根目录 Git-excluded `LOCAL_HARDWARE.md`。

### TD-07 校准 teacher 语言映射、prompt 与解码

依赖：TD-05、TD-06。

- [x] 为 18 路 v1 固定语言名称映射：`zho_Hans -> Chinese`、`zho_Hant -> Traditional Chinese`、`eng_Latn -> English`、`jpn_Jpan -> Japanese`、`kor_Hang -> Korean`；简体/繁体输出分别执行脚本合规检查。
- [x] 以官方“只输出翻译结果、不要额外解释”模板为起点，固定 prompt version、chat template、是否使用 system prompt、source/target 名称语言和输入分隔方式。
- [x] 在冻结的人类 dev/reference 小样本上比较 greedy/确定性解码与官方推荐采样参数，逐路由报告 chrF/SacreBLEU、脚本合规、额外解释、source copy、空输出和长度比。
- [x] 在查看完整 train 输出前选择唯一规范 decode profile；若采样模式无法跨 batch/resume 稳定重放，则不得作为规范 profile。
- [x] 为 18 个 v1 路由分别冻结最大输入/输出长度、stop 条件和异常阈值，防止某一路由用总体平均掩盖失败。
- [x] 对 prompt echo、额外解释、错语言/错脚本、繁体退化为简体、截断、重复、占位符损坏和 source copy 建立正反例测试。
- [x] 保存逐样本 teacher raw output 与 reference 对照；不得将 dev teacher output 混入 student train。
- [x] 保持 `Chinese` / `Traditional Chinese` 名称和既有 prompt/decode，不增加 locale-specific prompt；在冻结 human dev 上校准 `zho_Hans -> zho_Hant` 与反方向。
- [x] 为新增两路冻结长度、stop、异常阈值和路线专用 source-copy/脚本规则；共享汉字、数字、缩写、专名和合法不变短句不得被通用 source-copy 规则误杀。
- [x] 每路完成固定样本质量指标、人工检查与精确 replay；新 profile/addendum 身份不得回写 18 路 v1 校准报告。

产物：不可变 18 路 v1 校准、两条新增路线的校准 addendum、路线专用过滤测试和 20 路组合校准身份。

完成条件：20 个路由都有通过预设质量/格式门槛的唯一、可重放 teacher profile；任一新增路线失败都阻塞 TD-08。

v1 完成记录：TD-07 于 2026-07-15 完成 18 路校准。216 条冻结 dev 样本覆盖 18 路由；greedy 宏观 chrF 28.615981、char-SacreBLEU 33.923799、接受率 0.995370、脚本合规率 1.0，且 18 路无失败项。官方采样 chrF 仅高 0.014524，未达到 +2.0 切换门槛；两个 profile 的 36 条独立 replay 均逐字一致，最终冻结 `greedy-v1`，test 从未读取。该记录不包含新增两路。

### TD-08 生成 D0 smoke 并验收 D1 最小可用蒸馏数据

依赖：TD-05、TD-07。

- [x] 实现 v1 `scripts/generate_teacher_data.py`，只读取冻结 train source/`sample_group_id`，显式拒绝 dev/test，并按 18 个跨语言路由生成离散 UTF-8 teacher targets。
- [x] 支持 dry-run、确定性分片、原子 shard、逐样本 checkpoint/resume、缓存校验和中断恢复；worker/batch/resume 差异不得改变规范输出身份。
- [x] 每条记录保存 teacher revision/hash、运行后端、prompt version、decode config/seed、输入 sample/group ID、raw response、normalized target、raw/normalized hash 和生成 run manifest。
- [x] raw response 与 accepted target 分开保存；过滤空输出、额外解释/prompt echo、source copy、错语言/错脚本、异常长度、截断、重复和占位符损坏，并保留逐原因拒绝计数。
- [x] 每个路由至少人工检查 20 条 accepted 和 20 条 rejected（不足时全部），繁体目标额外抽检简繁混淆、地区词和共享汉字误判。
- [x] 输出 v1 18 路的输入数、成功数、拒绝率、重试率、长度/脚本/来源分布和 teacher 吞吐；任一路由低于冻结通过门槛时停止发布。
- [x] 使用相同 artifact/profile 对固定分片独立重放，验证 raw/normalized 输出和 manifest 身份符合 TD-07 的复现契约。
- [x] 发布有界 distilled train corpus 和完成 manifest；dev/test 继续只保留冻结的人类参考，teacher 从未消费 test。
- [x] 冻结独立 D1 配置/manifest 身份：沿用 D0 teacher/prompt/decode/filter，候选 source 为 D0 的确定性超集；每路由 2,224 个、总计 40,032，禁止覆盖 D0 目录或复用 D0 的 complete 身份冒充 D1。
- [x] 生成并过滤 D1；每路由 accepted 至少 2,000、总 accepted 至少 36,000，且接受率、脚本合规、重试、长度、source-copy、截断和 provenance 继续通过冻结门槛。
- [x] 对 D1 独立执行逐路由人工抽检、繁体/共享汉字专项检查和固定分片精确 replay；D0 的审查与 replay 只能作为管线先验证据，不能替代 D1 运行证据。
- [x] 原子发布 D1 v1 raw/accepted/filtered、质量报告、审查证明和 complete manifest；该 v1 曾满足 18 路门槛，但范围修正后不再单独具备 TD-15 输入资格。
- [x] 从新的 human composite train-only source 为 `zho_Hans -> zho_Hant` 与反方向各确定性选择 2,224 个候选；保持既有 `Chinese` / `Traditional Chinese` prompt，不修改或重放无关 18 路。
- [x] 对两条新增路线分别生成、过滤并达到至少 2,000 accepted；使用路线专用 source-copy、脚本、词汇差异、语义/实体/数字/占位符保持门槛。
- [x] 每路独立完成 accepted/rejected 人工审查与固定分片精确 replay，发布 raw/accepted/filtered、质量报告和 manifest-last addendum。
- [x] 发布引用 D1 v1 与新增两路 addendum 的 20 路 composite manifest；验证 v1 的 manifest/evidence/accepted 哈希保持不变，只有 composite 可供 TD-15 使用。

产物：不可变 D0/D1 v1、两路简繁互转 distilled addendum、20 路 composite、路线质量/人工审查/replay 和完整 provenance manifest。

完成条件：D0/D1 v1 证据保持有效，两条新增路线各至少 2,000 accepted，20 路 composite 通过质量、复现、许可/provenance 和 test 隔离验收后才能进入 TD-15。

阶段记录：D0 smoke 于 2026-07-15 完成。冻结 train 按 18 路由各生成 128 条，共 2,304 条；人工全检 381 条分层队列并剔除 39 条语义错误，受限恢复 4 条日中共享汉字 `source_copy` 误杀。最终接受 2,263 条、过滤 41 条，最低路由接受率 0.960938，全部路由脚本合规率 1.0、重试率 0、质量失败项为空。36 条独立 replay 的 raw/normalized 输出均精确一致；D0 complete manifest SHA-256 为 `2e0beb51e0b5020f7248da4d0f7bdd544bb0274c29c0efc22affa9d83ff1639e`，只作为 immutable smoke 证据。

阶段记录：D1 v1 于 2026-07-15 完成。独立配置从冻结 M0 train 选择 18 路由各 2,224 条，共 40,032 条，并逐字节验证后复用 D0 的每路由 128 条前缀（2,304 条）；其余 37,728 条由同一 GGUF Q8_0 teacher 新生成，生成墙钟 18,382.615340 秒。人工逐条检查 444 条队列，剔除 52 条自动接受的语义/实体/数字/意图错误，受限恢复 31 条有效共享汉字、数字、缩写和专名的 `source_copy` 误杀。最终接受 39,941 条、过滤 91 条；逐路由 accepted 为 2,211～2,223，最低接受率 0.994155、最低脚本合规率 0.999101、重试率全部为 0、质量失败项为空。独立重载 replay 的 36 条 raw/normalized 输出全部精确一致，dev/test 从未被 teacher 消费。D1 generation contract SHA-256 为 `2e54be92d270af3acac76251f25e31987a876f3e098dfb7bbbc73c696a470b1a`，complete manifest SHA-256 为 `9de9a4c251504c9ee157bec2dc4eefea8acd760d808672c15704f5c884b9ff2c`，tracked evidence 为 `artifacts/model-training/reports/teacher/distillation/d1.json`。重复 finalize 后七类产物哈希全部不变。

20 路完成记录：新增两路各生成 2,224 个候选，`zho_Hans->zho_Hant` 接受 2,213 条，`zho_Hant->zho_Hans` 接受 2,207 条；72 条人工队列全部审查，手工剔除 3 条大陆词汇残留，恢复 2 条与 human reference 精确一致的合法不变句。4 条独立 replay 的 raw/normalized 输出全部一致，质量失败项为空，dev/test 从未消费。addendum manifest SHA-256 为 `8700222adb328a4f7aac3dc92c46b53183dba7d1c46c97fd12e4d6eaab7a942f`；最终 composite 引用不可变 D1 v1 与 addendum，共 44,361 条、20 路，每路至少 2,207 条，manifest SHA-256 为 `fe72be6a588fda2a328e8c300d799061cab62ecfaabf13a702e637eb4dd8cd1e`。重复构建字节一致，TD-09 仍未启动。

### TD-09 实现编码、collator 与 student 构造

依赖：TD-01；实现可先使用 schema fixture，完整验收依赖 TD-05 冻结的全路由 fixture。

- [x] 只从 `artifacts/tokenizers/mvp-tokenizer-v0/` 离线加载 tokenizer，校验冻结 manifest SHA-256、49,152 稠密 ID、五个语言 token 和 fast backend。
- [x] 实现 source 编码与 target labels：source language token/`</s>`、target language token/`</s>`、padding mask 和 `-100` loss ignore index 均符合锁定 Transformers 语义。
- [x] 明确 source/target 最大长度、截断和丢弃策略，逐标签路由累计原始/截断 token 数；不得静默截断而不报告。
- [x] 实现方向感知 collator，拒绝空文本、同标签、allowlist 外标签、缺失目标 token 和词表越界；两条简繁互转路线必须作为合法方向通过。
- [x] 从配置创建 `mvp_e8_d2_v48k`，断言 shared/encoder/decoder embedding、`lm_head`、特殊 token、decoder start/generation config 与 tokenizer 完整一致且权重绑定。
- [x] 固定初始化种子并记录 state dict 身份；不得加载微型部署 checkpoint 或任何第三方权重。
- [x] 使用 20 路由 fixture 完成 CPU tokenize/collate/forward/backward 冒烟，并增加保存/离线重载测试。

产物：模型配置、数据集/编码/collator 模块、student builder 和自动化测试。

完成条件：所有 20 个路由均能产生正确输入/labels 和有限 loss，模型身份与冻结 tokenizer 可追溯；与 TD-05 一起关闭 M0。

完成记录：`scripts/mvp_student.py` 与 `scripts/validate_mvp_student.py` 已完成冻结 tokenizer 校验、方向感知编码/collator 和正式 student builder。TD-05 的 20 路 train composite CPU forward/backward loss 有限，双构造 state-dict SHA-256 均为 `66897f9c358802b9d39d66e61a8b39fad21236d11744b79df194c26db4da66a3`，离线保存/重载保持模型与 tokenizer 身份；机器记录为 `artifacts/model-training/reports/student/encoding-validation.json`，定向回归 `30 passed`。

### TD-10 实现训练循环、采样与运行记录

依赖：TD-09。

- [x] 实现 `scripts/train_mvp_model.py`，支持配置文件、dry-run、train/dev、固定 seed、设备/精度选择、梯度累积、gradient checkpointing、梯度裁剪和受控 dataloader worker。
- [x] 启动时探测设备、精度、总/可用内存和后端身份并写入 run manifest；训练代码不得按 GPU 型号、固定显存容量或盘符分支。
- [x] 从配置读取设备内存预算/预留/最大利用率、主机与 dataloader 内存预算、micro batch、累积、最大长度和 worker；有效设备内存上限取绝对预算、总容量乘最大利用率、总容量减预留三者的最小值，预算缺失或探测容量不足时明确失败。
- [x] 实现方向感知采样器，记录每个 batch/step 的路由组成、epoch、样本位置和实际 token 数；低资源方向权重必须来自冻结配置。
- [x] 固定 optimizer、scheduler、warmup、label smoothing（若使用）、最大 step/token 预算和验证频率，所有有效超参数进入配置哈希。
- [x] 记录 global/optimizer step、train/dev loss、学习率、梯度范数、tokens/s、样本/s、显存峰值、wall time、截断率和异常跳过数。
- [x] 对 NaN/Inf loss/gradient、OOM、空 batch、数据耗尽、配置/数据哈希变化明确失败；仅 TD-14 benchmark 模式可按配置的有限重试预算搜索候选，正式训练不得静默调参或继续发布。
- [x] checkpoint 选择只读取 dev 指标；训练脚本不得打开 test split。
- [x] 增加 CPU/小模型单步、梯度累积边界、采样重现、非法 loss、资源预算不足、profile 切换和训练日志 schema 测试，证明修改显存预算不需要改代码。

产物：可配置训练 CLI、方向采样器、结构化运行日志和自动化测试。

完成条件：fixture 上可稳定完成多个 optimizer step，日志能够从数据、配置和 seed 重建运行语义。

完成记录：正式 student 完成 2 optimizer step / 20 samples / 20 路各一次曝光，mean/final loss 为 `10.6890940666199` / `10.5333671569824`；两次独立运行的 loss、step、sampler state 与语义事件 trace 精确相同，trace SHA-256 为 `b937866624470c1764aacaab155690826eebb0f841d11159d1d83b0ef1236b74`。机器记录为 `artifacts/model-training/reports/student/training-smoke.json`，定向回归 `37 passed`。

### TD-11 实现原子 checkpoint 与精确恢复

依赖：TD-10。

- [x] checkpoint 保存模型、optimizer、scheduler、scaler、global step、epoch、已消费样本/token、梯度累积相位、采样器状态及 Python/NumPy/PyTorch CPU/CUDA RNG。
- [x] manifest 绑定数据/config/tokenizer/code/依赖哈希、Git commit/dirty 状态、设备/CUDA 环境、文件清单、大小、SHA-256 和 `status=complete`。
- [x] 使用同目录 staging、逐文件 fsync/校验和最终原子发布；拒绝不完整状态、文件缺失、哈希错误、路径穿越、符号链接和身份不匹配。
- [x] 支持显式 `--resume-from`，恢复后不得重复或跳过样本，不得重置 scheduler、累积相位或 RNG。
- [x] 做故障注入：写权重/optimizer/manifest 前后中断均不得发布半成品，旧 checkpoint 保持可用。
- [x] 在同一锁定环境比较 uninterrupted 与 resumed 短训练的 step、学习率、采样序列、loss 和权重；优先要求精确相等，若存在已证实的非确定性算子则预先冻结容差和说明。
- [x] 定义 checkpoint 保留/清理策略，任何删除只能发生在新 checkpoint 完整验证后。

产物：checkpoint/resume 模块、完整性验证器、故障注入与恢复一致性报告。

完成条件：从任一完整 checkpoint 恢复可重现连续训练语义；损坏或错配 checkpoint 被明确拒绝。

完成记录：正式 student 的连续 2 step 与 step 1 中断/恢复路径在 loss、step、学习率、sampler/RNG、语义 trace、model/optimizer/scheduler/scaler/gradient/RNG/trainer payload 上全部精确一致。机器记录为 `artifacts/model-training/reports/student/checkpoint-resume.json`。原定向回归为 `15 passed, 1 skipped`（文件 symlink 创建权限条件）；2026-07-16 已改用无需管理员权限的真实 NTFS directory junction/reparse point，并用定向测试覆盖 payload-link 拒绝分支，当前检查点回归 `9 passed`、全套离线回归 `196 passed`，无跳过。

### TD-12 完成 M1 小样本过拟合与恢复验收

依赖：TD-05、TD-11。

- [x] 使用正式 `mvp_e8_d2_v48k` 和固定 20 路由微型 fixture 建立随机初始化基线，在看训练结果前冻结最大 step/token 预算、解码配置和验收阈值。
- [x] 在冻结预算内将 fixture mean loss 降至初始基线的 10% 以下；每个有向路由至少一条固定记忆样例在 greedy/固定解码下得到正确目标语言和规范化 exact-match 目标文本。
- [x] 确认 20 个路由均被采样，任何路由饿死、错误目标语言或空输出都视为失败；source-copy 按跨语言/简繁互转各自合同判定。
- [x] 从中途 checkpoint 恢复并完成同样训练预算，与连续运行比较最终 step、采样、loss、权重和固定生成结果。
- [x] 完成最终 HF checkpoint 的离线保存/重载，验证 tokenizer 未被修改、模型词表仍为 49,152 且 generation config 完整。
- [x] 记录显存峰值、吞吐、耗时、loss 曲线和所有固定样例，不把过拟合结果描述为真实翻译质量。

产物：M1 过拟合 checkpoint、连续/恢复对照、生成回归和验收报告。

完成记录：冻结的 300-step 预算内 initial/final eval loss 比率为 `0.0145852494989926`，20 路各曝光 300 次且 20/20 生成 exact match；step 150 恢复与连续运行的全部训练 payload 精确一致。M1 HF state SHA-256 为 `3cfc2ba0d33afb05f5ec26b4a132f9b491548d58ab55ec13910da36ffabc8273`，机器记录为 `artifacts/model-training/reports/student/m1-overfit.json`。

完成条件：plan 的 M1 门槛全部满足；未通过前不得进入真实数据 GPU 配置冻结。

### TD-13 实现独立评测与方向汇总

依赖：TD-05、TD-09。

- [x] 实现 `scripts/evaluate_mvp_model.py`，离线加载数据、tokenizer 和 HF checkpoint；明确 dev/test 入口并默认拒绝在训练过程中读取 test。
- [x] 锁定 SacreBLEU/chrF 依赖、tokenization/signature、文本规范和生成参数；记录可复现版本与命令。
- [x] 报告 loss、SacreBLEU、chrF、目标脚本合规率、空输出率、source-copy 率、长度比、截断率和固定样例。
- [x] 先按 20 个标签路由输出明细，再汇总 12 个跨语言产品方向并单列 2 个简繁互转结果；中文汇总必须保留 `zho_Hans` 与 `zho_Hant` 明细和样本权重。
- [x] 对随机初始化、M1 过拟合和后续 M2 候选使用相同评测协议；禁止用训练内样本冒充 dev/test 质量。
- [x] 将逐样本输出、汇总 JSON、Markdown 报告和配置/模型/数据哈希关联，避免只保留不可追溯的终端文本。
- [x] 增加指标计算、脚本合规、20 路到 12+2 汇总、空 split、错标签和 test 访问边界测试。

产物：独立评测 CLI、指标/汇总模块、固定协议和自动化测试。

完成条件：任意合法 checkpoint 可在相同数据和生成配置下得到可复现的 20 路明细、12 个跨语言方向汇总与 2 个简繁互转结果。

完成记录：冻结 M1 checkpoint 在 200 条 dev 上完成独立评测，20/12/2 报告齐全；两次复放的逐样本、汇总、Markdown 与 manifest 字节一致。机器记录为 `artifacts/model-training/reports/student/evaluation-protocol.json`。

### TD-14 基准测试并冻结可配置训练资源 profile

依赖：TD-05、TD-12。

- [x] 探测当前执行主机的 accelerator/CPU、支持精度、后端/驱动、设备与主机内存，把真实硬件身份写入 benchmark/run manifest；语义配置不保存 GPU 型号。
- [x] 从配置读取设备/精度候选和设备内存预算、预留显存、最大利用率、主机/dataloader 内存预算及 OOM 重试上限；若回退精度或环境，记录理由和新 profile 身份。
- [x] 使用真实长度分布的 train/dev 小切片比较 micro batch、梯度累积、gradient checkpointing、最大 source/target 长度和 dataloader worker，候选组合不得来自隐藏常量。
- [x] 对每个候选记录峰值设备/主机内存、tokens/s、samples/s、step time、验证耗时、OOM/重试和截断率；不得只用短 synthetic 句估算正式容量。
- [x] 选择满足配置预算、安全余量、吞吐和截断门槛的唯一 M2 profile，冻结资源预算、optimizer/scheduler、batch、累积、长度、验证/checkpoint 频率和训练 token/step 预算。
- [x] 验证 checkpoint/staging/log 热路径使用配置的高吞吐运行根，最终发布遵守 TD-01 边界，不依赖盘符。
- [x] 完成至少 100 个 optimizer step 的 soak，期间至少执行 2 次 dev 验证和 2 次 checkpoint 发布，确认无显存持续增长、NaN/Inf、数据停顿或 checkpoint 阻塞。

产物：硬件基准报告、运行时硬件 manifest 和冻结的 M2 训练资源 profile。

完成条件：存在一个实测峰值不超过配置预算、满足预留内存约束、可恢复且数据截断可接受的唯一 M2 配置；自动化测试证明换显存容量只需调整 profile。

完成记录：基于 10,240 条真实长度样本选择 `mb64-ga2-l64-w2-no-gc`，冻结 profile SHA-256 `9384e5349839ebf5616ae6041e16343656ea0341fc0ddd305ef57590c686f47e`。100-step soak 完成两次验证/两次 checkpoint，峰值约 3.63 GB、零截断/异常，并从 step 100 恢复到 101。机器记录为 `artifacts/model-training/reports/m2/resources/profile.json`。

### TD-15 冻结蒸馏配方与等预算 A/B 契约

依赖：TD-05、TD-08、TD-13。

- [x] 只以 TD-08 的 20 路 distilled composite 与 TD-05 human composite 的 accepted 交集建立固定 A/B cohort；D0 或 D1 v1 单独禁止进入正式 A/B。teacher 失败/filtered source 必须从两组同时排除。
- [x] `human-only` 组对固定 cohort 使用人类 target，`distilled` 组对完全相同的 source/group ID 使用 Hy-MT2 7B teacher target；dev/test 两组都只使用冻结的人类参考。
- [x] 冻结两组相同的 student 初始 state-dict hash、source 样本顺序、路由权重、micro batch、梯度累积、optimizer/scheduler、最大 optimizer step 和 checkpoint/eval 频率。
- [x] 明确定义“等预算”为相同 source 曝光序列与 optimizer step 数；teacher target 与 human target 的长度差异单独报告，不得在看到结果后通过追加 step、样本或方向曝光补偿某一组。
- [x] 在训练前统计两组逐路由的样本数、source/target token、截断率、脚本合规和 target 差异，验证 20 个路由的 source 身份与曝光计划逐项一致。
- [x] 在配置中预先冻结 dev 选择规则与 tie-break：聚合 chrF/SacreBLEU、dev loss、目标脚本合规、空输出/source-copy 和逐路由最大允许退化均需有明确优先级或阈值；`zho_Hans` 与 `zho_Hant` 分开判定。
- [x] 生成两份不可变训练 recipe/manifest 和差异报告；除 target 文本及其 provenance/hash 外，任何影响优化预算的字段不同都应使 A/B 校验失败。
- [x] 用两组 recipe 分别完成短 dry-run，验证采样序列、初始权重、step 边界和评测入口一致，且训练代码无法访问 test。

产物：human-only/distilled 两份冻结 recipe、共同 cohort manifest、等预算校验器和训练前差异报告。

完成条件：两组只在训练 target 及其 provenance 上存在预期差异，比较预算和 dev 选择规则已在查看 M2 结果前冻结。

完成记录：严格交集含 44,313 条、20 路最少 2,207；两份 recipe 的 512 次 source 曝光和 sampler state 精确一致，只允许 target/provenance 与 arm 输入哈希差异。dev 规则已冻结且 TD-15 未访问 test；机器记录为 `artifacts/model-training/reports/m2/distillation-ab.json`。

### TD-16 训练并冻结基于合格语料能力的 MVP 模型（当前 suspended）

恢复依赖：TD-05 schema v4；任务组及 TD-16C～TD-16F 文件必须先按 ability-first mixed 配方重写。旧 TD-08/TD-15 只保留诊断证据。任务组见 [`td-16-m2-training.md`](../task/mvp-model-training/td-16-m2-training.md)。

- [x] 在44,313条共同 source 上完成 human-only/distilled 两臂 1,000-step 上限 A/B；只改变 target/provenance，正式 test 未访问。
- [x] 按冻结 dev 规则选择 human-only step 1,000；distilled 最佳为 step 900，但未通过总体 chrF、SacreBLEU 和逐路由退化门槛。
- [x] 记录结论：该 A/B 只诊断 teacher target 是否可替代 human target，不是完整 226,218 条 human M0 训练，也不是最终 MVP。
- [x] **TD-16A**：将性能优先训练器合并到主分支；缓存、预编码/worker、分桶、pinned/non-blocking、batch/累积、allocator、optimizer、日志和内存预算均配置化，本机 profile 不入 Git；重复训练按能力等价而非权重 hash 验收。
- [x] **TD-16B**：使用完整旧 M0 执行从零长训诊断；step 15k 后 early-stop，确认 226,218 条路由记录只有 11,411 个语义组，MASSIVE 本地化 target 不满足通用 MT 忠实度，所有 checkpoint 不准入后续阶段。
- [ ] **TD-16C**：TD-05 schema v4完成后，从零使用质量实收teacher/human corpus并按80/20 sampling weight训练60M，按预注册dev能力线和early-stop选择checkpoint。
- [ ] **TD-16D**：仅当 TD-16C 存在未过线弱路由时，对这些路由各新增 10,000 accepted target 并补训一次；若首轮已过线则记录为无需执行。
- [ ] **TD-16E**：重复运行最终配方，以能力容差和 time-to-quality 验证统计等价并冻结唯一候选。
- [ ] **TD-16F**：只对唯一候选执行一次正式 test，发布最终 HF MVP、20路报告和不可重放 receipt。

产物：不可变 A/B/旧 M0 长训诊断、硬件可配置训练器、schema v4 mixed corpus、60M 能力候选、可选弱路由 patch、重复训练等价报告、唯一最终 HF MVP 和一次性正式 test 报告。

完成条件：TD-16A～TD-16F 全部完成；schema v4 mixed 60M 达到预注册总体/逐路由能力线，唯一候选通过重复能力等价和冻结 dev 选择，test 只消费一次。A/B 候选、旧 M0 checkpoint、性能 soak、训练完成或单次 loss 下降均不能单独完成 TD-16。

### TD-17 完成 M3 CTranslate2 回接与量化诊断

依赖：TD-16F。

- [ ] 将现有 CTranslate2 验证逻辑泛化到训练后 HF checkpoint，创建新的模型训练部署记录，不覆盖已归档的随机 checkpoint 验收 JSON。
- [ ] 从本地候选生成 float32 诊断模型和 CPU INT8 验收模型，记录转换命令、版本、耗时、文件清单、大小和 SHA-256。
- [ ] 逐 ID 校验 frozen tokenizer、HF embedding/`lm_head`、float32 CT2 和 INT8 CT2 的 49,152 项词表及特殊 token。
- [ ] 对 20 个标签路由执行 source tokenize、`target_prefix`、去 prefix、decode 和固定样例推理；拒绝未知目标 token、错脚本、空输出和词表越界。
- [ ] 在查看 INT8 结果前冻结允许的逐路由/汇总指标退化容差；使用 TD-13 固定协议比较 HF、CT2 float32 和 CT2 INT8 的 chrF/SacreBLEU、脚本合规与固定样例，量化差异必须分路由报告，超出容差则停止验收并诊断。
- [ ] 记录 CPU 延迟、吞吐、compute type 和模型体积作为诊断值，不将单机短测宣称为生产性能。
- [ ] 生成独立 `tokenizer/` + `model/` 离线包，在新进程、离线标志、socket guard 和 manifest 校验下完成端到端推理。

产物：训练模型 CT2 float32/INT8 产物、离线包、转换/质量/性能诊断和自动化慢速测试。

完成条件：plan 的 M3 门槛全部满足；量化后仍保持语言控制和可解释的质量差异。

### TD-18 完成统一回归、文档与 review 准备

依赖：TD-01 至 TD-17。

- [ ] 整理人类数据、teacher artifact/生成数据、A/B recipe、student 模型、训练、评测、checkpoint、CT2 和离线包的配置/manifest/hash 关系，生成单一可追溯索引。
- [ ] 运行完整离线自动化测试和所有标记的慢速集成测试，记录命令、版本、测试数、耗时和结果；确认无运行产物或敏感/大体积数据被 Git 跟踪。
- [ ] 从干净临时目录验证 fixture 数据构建、teacher 离线 fixture 生成/过滤、M1 短训练/恢复、A/B recipe 校验、评测和离线 CT2 冒烟可重复执行。
- [ ] 更新 README、AGENTS、数据/训练/部署说明和已知限制；术语统一为 4 产品语言、5 标签、10 组关系、20 路、12 个跨语言方向 + 2 个简繁互转操作。
- [ ] 为 TD-01 至 TD-17 补齐输入、输出、验证命令、产物位置和完成证据，不创建相互矛盾的独立报告。
- [ ] 确认冻结 tokenizer 根哈希未变化，随机部署 checkpoint 未被描述为训练模型，M1 过拟合结果未被描述为真实质量。
- [ ] 准备统一 review 检查表，覆盖 teacher remote-code/离线边界、蒸馏 provenance 与 A/B 公平性、数据许可/泄漏、恢复正确性、质量边界、量化差异和部署风险。

产物：完整回归记录、工作流索引、更新文档和统一 review 输入。

完成条件：所有实现与运行证据齐全，可对整个 todo 和完整 task 集合执行一次统一 review。

## 完成条件

- [ ] TD-01～TD-15、TD-16A～TD-16F、TD-17～TD-18 全部完成，旧 M0/D1 诊断、schema v4 mixed corpus、M1、M2、M3 阶段门槛依次通过。
- [ ] `mvp-tokenizer-v0` 冻结根保持 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`，模型全链词表为 49,152 且 ID 顺序一致。
- [ ] MVP 数据覆盖 5 个标签桶、10 组无向模型关系和 20 个有向路由；简体、繁体分别有独立 dev/test，无 train/dev/test 泄漏。
- [ ] 锁定的 Hy-MT2 7B teacher 可离线重载并按固定 `Chinese` / `Traditional Chinese` prompt/decode 为 20 个 train 路由生成可审计的离散译文；v1 与新增两路 addendum/composite 均有完整 raw/accepted/provenance，teacher 未消费 test。
- [ ] M1 小样本过拟合、原子 checkpoint、故障拒绝和同环境恢复一致性通过。
- [ ] `mvp_e8_d2_v48k` 已完成旧共同 source A/B/长训诊断，并使用 TD-05 schema v4 的 80/20 mixed corpus 在 TD-16C 达到预注册能力线；最终配方通过重复能力等价后才选择唯一候选，test 只执行一次。
- [ ] 评测提供 20 路明细、12 个跨语言产品方向汇总与 2 个简繁互转结果，简体/繁体不被合并均值掩盖。
- [ ] 训练后 HF checkpoint 能离线重载并转换为 CT2 float32/CPU INT8，20 个路由全部完成离线推理回归。
- [ ] 完整测试、运行命令、版本、哈希、许可证、已知限制和失败恢复证据齐全。
- [ ] 文档不宣称 MVP 已达到生产翻译质量，不把全量 teacher 生成/在线 logits 蒸馏、200M 训练或生产性能混入本 todo。

## 统一 review 与归档

- [ ] TD-01～TD-15、TD-16A～TD-16F、TD-17～TD-18 全部完成后，对本 todo 和完整 task 集合执行一次统一 review；不为单个 TD 提前创建 review。
- [ ] review 通过后，将 todo、task 集合和 review 记录分别归档到 `work/done/`，并更新 plan/AGENTS 中的状态和相对链接。

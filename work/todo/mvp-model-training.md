# todo: MVP model training

状态：todo / active

## 来源

- plan：[MVP model training](../plan/mvp-model-training.md)
- task 索引：[MVP model training tasks](../task/mvp-model-training/index.md)
- 项目语言与方向口径：[README](../../README.md#语言与方向口径)
- 冻结 tokenizer：[mvp-tokenizer-v0](../../artifacts/tokenizers/mvp-tokenizer-v0/)
- tokenizer review：[mvp tokenizer review](../done/review/mvp-tokenizer.md)
- 部署兼容性 review：[CTranslate2 deployment review](../done/review/ctranslate2-deployment.md)

## 目标

使用不可变的 `mvp-tokenizer-v0`、锁定的 Hy-MT2 7B teacher 和从零初始化的 `mvp_e8_d2_v48k`，建立“有界人类平行数据 -> 离线 sequence-level 蒸馏 -> human-only/distilled 等预算训练与恢复 -> 独立评测 -> CTranslate2 float32/INT8 离线推理”的第一个真实模型闭环。

本 todo 包含有界 Hy-MT2 7B 蒸馏试点，但不包含全量 teacher 数据生成、在线 logits/hidden-state 蒸馏、`e12-d3`、约 200M 模型、生产质量门槛或移动端性能优化。

## 固定口径

- 产品语言为中文、英文、日文、韩文 4 种；模型标签为 `eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang` 5 个。
- MVP 数据必须覆盖 9 组无向平行语料；交换 source/target 后形成 18 个有向训练路由，并汇总为 12 个产品翻译方向。
- `zho_Hans <-> zho_Hant` 是简繁转换，不进入训练、评测或部署验收。
- 繁体可以少于简体，但必须有原生繁体样本和独立 dev/test；转换数据只能作为显式标注的增强数据，不能替代原生繁体验收。
- student 只使用 49,152 词表的 `mvp_e8_d2_v48k`，从零初始化，不加载任何第三方模型权重或随机部署验证 checkpoint。
- teacher 固定为锁定 revision/hash 的官方 Hy-MT2 7B artifact，只生成离散 UTF-8 译文；teacher 不进入 student 训练图，student 不继承其 tokenizer、权重或架构。
- 正式蒸馏 corpus 只从 train source 生成；仅允许在冻结的有界 human dev 子集上运行 teacher 以校准 prompt/decode，校准输出不得进入 student train，test 不得送入 teacher。
- test 只在最终候选冻结后执行一次正式评测；训练配置和 checkpoint 选择只使用 train/dev。

## 依赖关系

```text
TD-01
├─ 人类数据链：TD-02 -> TD-03 -> TD-04 -> TD-05
├─ teacher 运行时：TD-06
└─ student 基础链：TD-09 -> TD-10 -> TD-11

TD-05 + TD-06 -> TD-07 -> TD-08
TD-05 + TD-11 -> TD-12
TD-05 + TD-09 -> TD-13
TD-05 + TD-12 -> TD-14
TD-05 + TD-08 + TD-13 -> TD-15
TD-08 + TD-12 + TD-13 + TD-14 + TD-15 -> TD-16
TD-16 -> TD-17 -> TD-18
```

TD-09 至 TD-11 可在人类数据链构建期间先用 TD-01 的 schema fixture 开发，但涉及全路由和阶段验收时必须消费 TD-05 冻结的 fixture/manifest。M0 在 TD-05 与 TD-09 都完成后关闭；D0 在 TD-08 完成后关闭；M1 在 TD-12 完成后关闭；M2 在 TD-16 完成后关闭；M3 在 TD-17 完成后关闭。TD-18 负责整个 todo 的统一回归与 review 准备。

## 待办

### TD-01 冻结执行契约、目录与 Git 边界

依赖：无。

- [ ] 定义规范平行样本 schema，至少包含 `sample_id`、`sample_group_id`、`source_id`、`source_version`、`license`、`src_lang`、`tgt_lang`、`source_text`、`target_text`、`split`；teacher/转换增强样本增加生成 provenance。
- [ ] 固定 5 个允许标签、9 组允许的无向标签对、18 个有向路由，并明确拒绝同标签路由、简繁互转和 allowlist 外标签。
- [ ] 固定模型数据目录：`data/model/raw/`、`cache/`、`interim/`、`corpus/mvp/`、`reports/`；所有大体积数据默认 Git-ignored，只提交 schema、配置、lock、fixture 和精简报告。
- [ ] 固定训练运行目录和发布边界：热 checkpoint/staging 可配置到 SSD，完整校验后发布 Git-ignored HF/CT2 产物；提交内只保存配置、manifest、指标和文档。
- [ ] 定义 `configs/mvp_model_data.yaml`、`configs/mvp_e8_d2_v48k.yaml` 的字段、schema version、稳定序列化和配置哈希规则。
- [ ] 为 schema、方向矩阵、路径边界、未知字段、缺失字段和非法路由增加配置级自动化测试。

产物：数据/训练配置骨架、目录与 Git 边界、schema/方向矩阵测试。

完成条件：同一配置可以唯一确定允许的数据形态、模型身份和产物位置；未触碰冻结 tokenizer。

### TD-02 调研并锁定有界平行数据来源

依赖：TD-01。

- [ ] 针对 9 组无向标签对调研可下载版本、语言/脚本标注、许可证、数据卡、对齐质量、规模和获取方式，优先使用许可清晰的人类平行语料。
- [ ] 对繁体相关的 3 组语料逐一确认繁体侧为原生 `zho_Hant`，不把简转繁、粤语 `yue_Hant` 或脚本未知的中文静默归类为普通话繁体。
- [ ] 若某组缺少足够的人类平行语料，设计有界 synthetic 补充方案：必须保留原生文本侧、teacher 身份、prompt、解码参数和生成 manifest；不得因此引入大规模蒸馏范围。
- [ ] 为每组确定 MVP 的 train/dev/test 最小样本预算、扫描上限和下载上限；繁体可低于简体，但 dev/test 不得为空。
- [ ] 生成来源 registry 和 `configs/mvp_model_data.lock.json`，锁定 URI、版本、文件大小、SHA-256、许可证和逻辑处理顺序。
- [ ] 记录许可证不兼容、用途不明或无法稳定版本化的候选并排除，不以“能下载”代替可用性结论。

产物：`docs/model-training-dataset-research.md`、数据 registry、source lock 和 9 组覆盖矩阵。

完成条件：9 组语料均有明确、可审计的 MVP 来源方案；任何未关闭的来源或许可缺口都会阻塞 TD-03 正式构建。

### TD-03 实现确定性平行数据构建管线

依赖：TD-01、TD-02。

- [ ] 实现 `scripts/prepare_model_data.py` CLI 和独立的 `scripts/model_data_pipeline.py` 核心模块，保持仓库现有扁平模块结构。
- [ ] 支持 dry-run、source lock 校验、下载/断点续传、缓存复用、离线重建和失败后安全恢复；不得在正式构建时隐式解析 `latest`。
- [ ] 将不同来源解析成规范样本，使用稳定来源身份和规范内容生成 `sample_id`/`sample_group_id`，禁止 Python 内置 `hash()`、绝对路径和时间戳进入内容身份。
- [ ] 实现保守清洗：Unicode/空白规范、空文本、控制字符、HTML 残留、错误脚本占优、长度/长度比和异常内容过滤；禁止小写化、简繁转换、假名转换或韩文转写。
- [ ] 对原生、人工平行、teacher synthetic 和脚本转换增强数据使用不同 provenance，不允许清洗过程丢失来源类型。
- [ ] 输出规范 UTF-8/LF JSONL、拒绝原因统计、来源/标签对计数和原子 manifest；manifest 最后发布且逐文件记录大小/SHA-256。
- [ ] 用小型 fixture 覆盖所有来源适配器、错误路径、缓存损坏、网络失败和半成品清理。

产物：模型数据构建 CLI/核心模块、fixture、manifest 和自动化测试。

完成条件：从已锁定缓存可完全离线重建相同规范样本；失败不会发布可被误认为完成的 corpus。

### TD-04 实现分组切分、去重与泄漏防护

依赖：TD-03。

- [ ] 在扩展正反方向前按无向平行关系生成稳定 group；同一对齐关系、反向样本、同文档片段和已知派生样本必须进入同一个 split。
- [ ] 在规范文本、source、target 和 pair 层执行 exact 去重；对 train/dev/test 执行跨集合 near-duplicate/污染检查并记录参数与命中原因。
- [ ] split 使用稳定 group hash 和版本化比例生成，禁止逐行随机拆分；test 身份在数据构建阶段冻结。
- [ ] 与 tokenizer corpus/holdout、正式 MT 评测集和同一数据源重复版本进行可追溯污染检查，不把 tokenizer holdout 当成模型质量 test。
- [ ] 在 split 完成后扩展正反方向，验证 18 个路由中任一方向都不会把对应反向关系泄漏到其他 split。
- [ ] 验证 worker 数、缓存命中、输入完成顺序和 fresh/resume 路径不会改变 corpus、split 或 manifest 字节。
- [ ] 增加反向泄漏、跨 split 近重复、派生样本、错误 group 和非确定性顺序的失败测试。

产物：确定性 split/dedup/leakage 模块、污染报告和自动化测试。

完成条件：train/dev/test 在 group 层相互隔离；两次独立构建的规范 corpus 与确定性 manifest 字节级一致。

### TD-05 构建并验收 M0 数据集

依赖：TD-04。

- [ ] 建立 `tests/fixtures/model_data/` 微型数据，9 组无向标签对均有样本，扩展后完整覆盖 18 个有向路由及非法路由反例。
- [ ] 构建有界真实 MVP corpus，确认 5 个标签桶、9 组无向语料和 18 个有向路由均非空；简体、繁体分别具有独立 dev/test。
- [ ] 固定方向采样策略，报告原始样本数、过滤后样本数、正反扩展数、训练权重和有效曝光；禁止低资源方向无界重复。
- [ ] 报告每个标签对/split 的来源占比、原生/synthetic/增强占比、长度与长度比分布、脚本合规率和过滤原因。
- [ ] 对每组执行分层人工抽检：至少检查 20 条 accepted train、10 条 accepted dev/test 和 20 条 rejected（不足时检查全部），覆盖长度边界、繁体与混合脚本样本，并冻结精简审查记录。
- [ ] 使用不同 worker/cache 状态完成两次独立构建，逐文件比较 corpus、manifest 和确定性报告 SHA-256。
- [ ] 生成 M0 验收报告；任何标签对为空、繁体 dev/test 缺失、泄漏、来源不明或复现失败都阻塞训练数据发布。

产物：有界 MVP 模型 corpus、18 路由 fixture、质量/覆盖/复现报告和完成 manifest。

完成条件：plan 的 M0 数据与编码前置条件中“数据”部分全部满足，数据集被标记为可供训练链消费。

### TD-06 锁定并验证 Hy-MT2 7B teacher 运行时

依赖：TD-01。

- [ ] 锁定腾讯官方 [`tencent/Hy-MT2-7B`](https://huggingface.co/tencent/Hy-MT2-7B) 或经验证的官方同模型运行 artifact，记录 Hugging Face revision、模型/代码/chat template/许可证文件清单、大小和 SHA-256。
- [ ] 记录官方 [Apache-2.0 许可证](https://huggingface.co/tencent/Hy-MT2-7B/blob/main/LICENSE.txt)，并明确模型许可证不自动解决输入语料或生成数据的权利边界。
- [ ] 审查并锁定官方示例要求的 `trust_remote_code` 内容；正式生成只从本地固定快照加载，启用离线标志和网络阻断，不执行浮动 `main` 或运行时下载。
- [ ] 为 teacher 建立与 student 依赖隔离或明确兼容的运行 profile，锁定 Python、Transformers、PyTorch、CUDA/后端和启动命令，不让 teacher 依赖改写 student 主环境。
- [ ] 在 RTX 4060 Ti 16 GB/CPU 上比较可行的官方 BF16 offload、FP8 或 GGUF 等运行路径；只选择官方来源且通过参考集验证的 artifact，不使用来源不明的社区量化。
- [ ] 对 5 个项目标签完成最小离线推理，验证官方支持的 Chinese、Traditional Chinese、English、Japanese、Korean 均能生成非空结果。
- [ ] 记录加载峰值内存/显存、单样本延迟、吞吐、输出稳定性和已知限制；若无可接受运行路径，D0 阻塞，不降级为其他 teacher。

产物：teacher artifact lock、remote-code 审查记录、离线运行 profile 和五标签冒烟报告。

完成条件：固定 teacher artifact 可在完全离线环境重载并完成五标签推理，所有执行代码和文件身份可审计。

### TD-07 校准 teacher 语言映射、prompt 与解码

依赖：TD-05、TD-06。

- [ ] 固定语言名称映射：`zho_Hans -> Chinese`、`zho_Hant -> Traditional Chinese`、`eng_Latn -> English`、`jpn_Jpan -> Japanese`、`kor_Hang -> Korean`；简体/繁体输出分别执行脚本合规检查。
- [ ] 以官方“只输出翻译结果、不要额外解释”模板为起点，固定 prompt version、chat template、是否使用 system prompt、source/target 名称语言和输入分隔方式。
- [ ] 在冻结的人类 dev/reference 小样本上比较 greedy/确定性解码与官方推荐采样参数，逐路由报告 chrF/SacreBLEU、脚本合规、额外解释、source copy、空输出和长度比。
- [ ] 在查看完整 train 输出前选择唯一规范 decode profile；若采样模式无法跨 batch/resume 稳定重放，则不得作为规范 profile。
- [ ] 为 18 个路由分别冻结最大输入/输出长度、stop 条件和异常阈值，防止某一路由用总体平均掩盖失败。
- [ ] 对 prompt echo、额外解释、错语言/错脚本、繁体退化为简体、截断、重复、占位符损坏和 source copy 建立正反例测试。
- [ ] 保存逐样本 teacher raw output 与 reference 对照；不得将 dev teacher output 混入 student train。

产物：teacher 语言映射、prompt/decode 配置、18 路由校准报告和输出过滤测试。

完成条件：18 个路由都有通过预设质量/格式门槛的唯一、可重放 teacher profile；失败路由阻塞 TD-08。

### TD-08 生成并验收有界 sequence-level 蒸馏数据

依赖：TD-05、TD-07。

- [ ] 实现 `scripts/generate_teacher_data.py`，只读取冻结 train source/`sample_group_id`，显式拒绝 dev/test，并按 18 个路由生成离散 UTF-8 teacher targets。
- [ ] 支持 dry-run、确定性分片、原子 shard、逐样本 checkpoint/resume、缓存校验和中断恢复；worker/batch/resume 差异不得改变规范输出身份。
- [ ] 每条记录保存 teacher revision/hash、运行后端、prompt version、decode config/seed、输入 sample/group ID、raw response、normalized target、raw/normalized hash 和生成 run manifest。
- [ ] raw response 与 accepted target 分开保存；过滤空输出、额外解释/prompt echo、source copy、错语言/错脚本、异常长度、截断、重复和占位符损坏，并保留逐原因拒绝计数。
- [ ] 每个路由至少人工检查 20 条 accepted 和 20 条 rejected（不足时全部），繁体目标额外抽检简繁混淆、地区词和共享汉字误判。
- [ ] 输出 18 路由的输入数、成功数、拒绝率、重试率、长度/脚本/来源分布和 teacher 吞吐；任一路由低于冻结通过门槛时停止发布。
- [ ] 使用相同 artifact/profile 对固定分片独立重放，验证 raw/normalized 输出和 manifest 身份符合 TD-07 的复现契约。
- [ ] 发布有界 distilled train corpus 和完成 manifest；dev/test 继续只保留冻结的人类参考，teacher 从未消费 test。

产物：有界 Hy-MT2 7B raw/accepted/filtered 数据、生成器、18 路由质量报告和完整 provenance manifest。

完成条件：plan 的 D0 门槛全部满足；只有通过质量、复现、许可/provenance 和 test 隔离验收的 teacher targets 才能进入 TD-15。

### TD-09 实现编码、collator 与 student 构造

依赖：TD-01；实现可先使用 schema fixture，完整验收依赖 TD-05 冻结的全路由 fixture。

- [ ] 只从 `artifacts/tokenizers/mvp-tokenizer-v0/` 离线加载 tokenizer，校验冻结 manifest SHA-256、49,152 稠密 ID、五个语言 token 和 fast backend。
- [ ] 实现 source 编码与 target labels：source language token/`</s>`、target language token/`</s>`、padding mask 和 `-100` loss ignore index 均符合锁定 Transformers 语义。
- [ ] 明确 source/target 最大长度、截断和丢弃策略，逐标签路由累计原始/截断 token 数；不得静默截断而不报告。
- [ ] 实现方向感知 collator，拒绝空文本、同标签、简繁互转、allowlist 外标签、缺失目标 token 和词表越界。
- [ ] 从配置创建 `mvp_e8_d2_v48k`，断言 shared/encoder/decoder embedding、`lm_head`、特殊 token、decoder start/generation config 与 tokenizer 完整一致且权重绑定。
- [ ] 固定初始化种子并记录 state dict 身份；不得加载微型部署 checkpoint 或任何第三方权重。
- [ ] 使用 18 路由 fixture 完成 CPU tokenize/collate/forward/backward 冒烟，并增加保存/离线重载测试。

产物：模型配置、数据集/编码/collator 模块、student builder 和自动化测试。

完成条件：所有 18 个路由均能产生正确输入/labels 和有限 loss，模型身份与冻结 tokenizer 可追溯；与 TD-05 一起关闭 M0。

### TD-10 实现训练循环、采样与运行记录

依赖：TD-09。

- [ ] 实现 `scripts/train_mvp_model.py`，支持配置文件、dry-run、train/dev、固定 seed、设备/精度选择、梯度累积、gradient checkpointing、梯度裁剪和受控 dataloader worker。
- [ ] 实现方向感知采样器，记录每个 batch/step 的路由组成、epoch、样本位置和实际 token 数；低资源方向权重必须来自冻结配置。
- [ ] 固定 optimizer、scheduler、warmup、label smoothing（若使用）、最大 step/token 预算和验证频率，所有有效超参数进入配置哈希。
- [ ] 记录 global/optimizer step、train/dev loss、学习率、梯度范数、tokens/s、样本/s、显存峰值、wall time、截断率和异常跳过数。
- [ ] 对 NaN/Inf loss/gradient、OOM、空 batch、数据耗尽、配置/数据哈希变化明确失败；不得静默跳过并继续发布候选。
- [ ] checkpoint 选择只读取 dev 指标；训练脚本不得打开 test split。
- [ ] 增加 CPU/小模型单步、梯度累积边界、采样重现、非法 loss 和训练日志 schema 测试。

产物：可配置训练 CLI、方向采样器、结构化运行日志和自动化测试。

完成条件：fixture 上可稳定完成多个 optimizer step，日志能够从数据、配置和 seed 重建运行语义。

### TD-11 实现原子 checkpoint 与精确恢复

依赖：TD-10。

- [ ] checkpoint 保存模型、optimizer、scheduler、scaler、global step、epoch、已消费样本/token、梯度累积相位、采样器状态及 Python/NumPy/PyTorch CPU/CUDA RNG。
- [ ] manifest 绑定数据/config/tokenizer/code/依赖哈希、Git commit/dirty 状态、设备/CUDA 环境、文件清单、大小、SHA-256 和 `status=complete`。
- [ ] 使用同目录 staging、逐文件 fsync/校验和最终原子发布；拒绝不完整状态、文件缺失、哈希错误、路径穿越、符号链接和身份不匹配。
- [ ] 支持显式 `--resume-from`，恢复后不得重复或跳过样本，不得重置 scheduler、累积相位或 RNG。
- [ ] 做故障注入：写权重/optimizer/manifest 前后中断均不得发布半成品，旧 checkpoint 保持可用。
- [ ] 在同一锁定环境比较 uninterrupted 与 resumed 短训练的 step、学习率、采样序列、loss 和权重；优先要求精确相等，若存在已证实的非确定性算子则预先冻结容差和说明。
- [ ] 定义 checkpoint 保留/清理策略，任何删除只能发生在新 checkpoint 完整验证后。

产物：checkpoint/resume 模块、完整性验证器、故障注入与恢复一致性报告。

完成条件：从任一完整 checkpoint 恢复可重现连续训练语义；损坏或错配 checkpoint 被明确拒绝。

### TD-12 完成 M1 小样本过拟合与恢复验收

依赖：TD-05、TD-11。

- [ ] 使用正式 `mvp_e8_d2_v48k` 和固定 18 路由微型 fixture 建立随机初始化基线，在看训练结果前冻结最大 step/token 预算、解码配置和验收阈值。
- [ ] 在冻结预算内将 fixture mean loss 降至初始基线的 10% 以下；每个有向路由至少一条固定记忆样例在 greedy/固定解码下得到正确目标语言和规范化 exact-match 目标文本。
- [ ] 确认 18 个路由均被采样，任何路由饿死、错误目标语言、空输出或 source copy 异常都视为失败。
- [ ] 从中途 checkpoint 恢复并完成同样训练预算，与连续运行比较最终 step、采样、loss、权重和固定生成结果。
- [ ] 完成最终 HF checkpoint 的离线保存/重载，验证 tokenizer 未被修改、模型词表仍为 49,152 且 generation config 完整。
- [ ] 记录显存峰值、吞吐、耗时、loss 曲线和所有固定样例，不把过拟合结果描述为真实翻译质量。

产物：M1 过拟合 checkpoint、连续/恢复对照、生成回归和验收报告。

完成条件：plan 的 M1 门槛全部满足；未通过前不得进入真实数据 GPU 配置冻结。

### TD-13 实现独立评测与方向汇总

依赖：TD-05、TD-09。

- [ ] 实现 `scripts/evaluate_mvp_model.py`，离线加载数据、tokenizer 和 HF checkpoint；明确 dev/test 入口并默认拒绝在训练过程中读取 test。
- [ ] 锁定 SacreBLEU/chrF 依赖、tokenization/signature、文本规范和生成参数；记录可复现版本与命令。
- [ ] 报告 loss、SacreBLEU、chrF、目标脚本合规率、空输出率、source-copy 率、长度比、截断率和固定样例。
- [ ] 先按 18 个标签路由输出明细，再按 12 个产品方向汇总；中文汇总必须同时保留 `zho_Hans` 与 `zho_Hant` 明细和样本权重。
- [ ] 对随机初始化、M1 过拟合和后续 M2 候选使用相同评测协议；禁止用训练内样本冒充 dev/test 质量。
- [ ] 将逐样本输出、汇总 JSON、Markdown 报告和配置/模型/数据哈希关联，避免只保留不可追溯的终端文本。
- [ ] 增加指标计算、脚本合规、18->12 汇总、空 split、错标签和 test 访问边界测试。

产物：独立评测 CLI、指标/汇总模块、固定协议和自动化测试。

完成条件：任意合法 checkpoint 可在相同数据和生成配置下得到可复现的 18 路由明细与 12 方向汇总。

### TD-14 基准测试并冻结 RTX 4060 Ti 训练配置

依赖：TD-05、TD-12。

- [ ] 在本机 RTX 4060 Ti 16 GB 上验证 BF16 可用性与锁定 CUDA/PyTorch 稳定性；若回退精度或环境，记录理由和新依赖身份。
- [ ] 使用真实长度分布的 train/dev 小切片比较 micro batch、梯度累积、gradient checkpointing、最大 source/target 长度和 dataloader worker。
- [ ] 对每个候选记录峰值显存、tokens/s、samples/s、step time、验证耗时、OOM/重试和截断率；不得只用短 synthetic 句估算正式容量。
- [ ] 选择满足显存安全余量、吞吐和截断门槛的唯一 M2 profile，冻结 optimizer/scheduler、batch、累积、长度、验证/checkpoint 频率和训练 token/step 预算。
- [ ] 验证 checkpoint 写入位置不会让 E: 机械盘成为训练热路径；SSD staging 和最终发布遵守 TD-01 边界。
- [ ] 完成至少 100 个 optimizer step 的 soak，期间至少执行 2 次 dev 验证和 2 次 checkpoint 发布，确认无显存持续增长、NaN/Inf、数据停顿或 checkpoint 阻塞。

产物：硬件基准报告和冻结的 M2 本机训练 profile。

完成条件：存在一个在 16 GB 显存内稳定运行、可恢复且数据截断可接受的唯一 M2 配置。

### TD-15 冻结蒸馏配方与等预算 A/B 契约

依赖：TD-05、TD-08、TD-13。

- [ ] 以 TD-08 accepted teacher targets 与 TD-05 human references 的交集建立固定 A/B cohort；teacher 生成失败或被过滤的 source/group 必须从两组同时排除，不得只给 distilled 组补样或回退 human target。
- [ ] `human-only` 组对固定 cohort 使用人类 target，`distilled` 组对完全相同的 source/group ID 使用 Hy-MT2 7B teacher target；dev/test 两组都只使用冻结的人类参考。
- [ ] 冻结两组相同的 student 初始 state-dict hash、source 样本顺序、路由权重、micro batch、梯度累积、optimizer/scheduler、最大 optimizer step 和 checkpoint/eval 频率。
- [ ] 明确定义“等预算”为相同 source 曝光序列与 optimizer step 数；teacher target 与 human target 的长度差异单独报告，不得在看到结果后通过追加 step、样本或方向曝光补偿某一组。
- [ ] 在训练前统计两组逐路由的样本数、source/target token、截断率、脚本合规和 target 差异，验证 18 个路由的 source 身份与曝光计划逐项一致。
- [ ] 在配置中预先冻结 dev 选择规则与 tie-break：聚合 chrF/SacreBLEU、dev loss、目标脚本合规、空输出/source-copy 和逐路由最大允许退化均需有明确优先级或阈值；`zho_Hans` 与 `zho_Hant` 分开判定。
- [ ] 生成两份不可变训练 recipe/manifest 和差异报告；除 target 文本及其 provenance/hash 外，任何影响优化预算的字段不同都应使 A/B 校验失败。
- [ ] 用两组 recipe 分别完成短 dry-run，验证采样序列、初始权重、step 边界和评测入口一致，且训练代码无法访问 test。

产物：human-only/distilled 两份冻结 recipe、共同 cohort manifest、等预算校验器和训练前差异报告。

完成条件：两组只在训练 target 及其 provenance 上存在预期差异，比较预算和 dev 选择规则已在查看 M2 结果前冻结。

### TD-16 执行 M2 human-only/distilled 等预算训练

依赖：TD-05、TD-08、TD-12、TD-13、TD-14、TD-15。

- [ ] 在启动前验证共同 cohort、两份 recipe、corpus/teacher manifest、tokenizer、模型/训练配置、代码与依赖哈希，工作树状态和运行命令写入各自 run manifest。
- [ ] 从同一初始 state-dict hash 分别启动 human-only 与 distilled 两组 `mvp_e8_d2_v48k` 训练，使用冻结 M2 profile、相同 source 曝光和 optimizer step 预算；不得在运行中改变采样、超参数或给任一组追加预算。
- [ ] 按相同频率执行两组 dev loss/生成评测和原子 checkpoint；各组内部 checkpoint 及最终组间候选选择都只依据 TD-15 预先冻结的 dev 指标、逐路由退化阈值与 tie-break。
- [ ] 监控 NaN/Inf、OOM、方向曝光、截断率、吞吐和显存；异常恢复必须从已验证 checkpoint 继续并记录中断边界。
- [ ] 训练结束后分别冻结两组最佳 dev checkpoint，离线重载并验证权重、配置、tokenizer ID 空间和固定 dev 生成，再生成逐路由和聚合 A/B 对照。
- [ ] 严格按冻结规则选出唯一最终候选；若 distilled 未优于 human-only 或触发任一路由退化红线，则选择 human-only、记录蒸馏负结果并停止扩大 teacher 生成规模。
- [ ] 唯一候选冻结后只对该候选运行一次正式 test，生成 18 路由明细、12 产品方向汇总及随机初始化基线对照；不得为了比较两组而提前或重复读取 test。
- [ ] 验证两组 train/dev loss 全程有限，最终 dev loss 均低于同协议的随机初始化基线；任一运行异常结束必须明确标为失败，不得用另一组的成功掩盖。
- [ ] 明确记录空/弱方向、繁体差距和已知限制，不把“loss 下降”单独描述为可发布翻译质量。

产物：两组 M2 HF checkpoint 与运行/恢复记录、等预算 A/B 报告、唯一 dev 选择记录和一次性正式 test 报告。

完成条件：plan 的 M2 门槛全部满足；human-only/distilled 比较未使用 test，唯一候选完全由冻结 dev 规则选出且身份可追溯。

### TD-17 完成 M3 CTranslate2 回接与量化诊断

依赖：TD-16。

- [ ] 将现有 CTranslate2 验证逻辑泛化到训练后 HF checkpoint，创建新的模型训练部署记录，不覆盖已归档的随机 checkpoint 验收 JSON。
- [ ] 从本地候选生成 float32 诊断模型和 CPU INT8 验收模型，记录转换命令、版本、耗时、文件清单、大小和 SHA-256。
- [ ] 逐 ID 校验 frozen tokenizer、HF embedding/`lm_head`、float32 CT2 和 INT8 CT2 的 49,152 项词表及特殊 token。
- [ ] 对 18 个标签路由执行 source tokenize、`target_prefix`、去 prefix、decode 和固定样例推理；拒绝未知目标 token、错脚本、空输出和词表越界。
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
- [ ] 更新 README、AGENTS、数据/训练/部署说明和已知限制；术语继续使用 4 产品语言、5 标签、9 组语料、18 路由、12 产品方向。
- [ ] 为 TD-01 至 TD-17 补齐输入、输出、验证命令、产物位置和完成证据，不创建相互矛盾的独立报告。
- [ ] 确认冻结 tokenizer 根哈希未变化，随机部署 checkpoint 未被描述为训练模型，M1 过拟合结果未被描述为真实质量。
- [ ] 准备统一 review 检查表，覆盖 teacher remote-code/离线边界、蒸馏 provenance 与 A/B 公平性、数据许可/泄漏、恢复正确性、质量边界、量化差异和部署风险。

产物：完整回归记录、工作流索引、更新文档和统一 review 输入。

完成条件：所有实现与运行证据齐全，可对整个 todo 和完整 task 集合执行一次统一 review。

## 完成条件

- [ ] TD-01 至 TD-18 全部完成，M0、D0、M1、M2、M3 阶段门槛依次通过。
- [ ] `mvp-tokenizer-v0` 冻结根保持 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`，模型全链词表为 49,152 且 ID 顺序一致。
- [ ] MVP 数据覆盖 5 个标签桶、9 组无向平行语料和 18 个有向路由；简体、繁体分别有独立 dev/test，无 train/dev/test 泄漏。
- [ ] 锁定的 Hy-MT2 7B teacher 可离线重载并按固定 prompt/decode 为 18 个 train 路由生成可审计的离散译文；teacher 未消费 test，raw/accepted 数据与 provenance 完整。
- [ ] M1 小样本过拟合、原子 checkpoint、故障拒绝和同环境恢复一致性通过。
- [ ] `mvp_e8_d2_v48k` 完成人类 target 与 Hy-MT2 7B target 的等预算 M2 A/B；唯一候选只由冻结 dev 规则选择，若蒸馏无收益则记录负结果且不扩量，test 只执行一次正式评测。
- [ ] 评测提供 18 路由明细和 12 产品方向汇总，简体/繁体不被合并均值掩盖。
- [ ] 训练后 HF checkpoint 能离线重载并转换为 CT2 float32/CPU INT8，18 个路由全部完成离线推理回归。
- [ ] 完整测试、运行命令、版本、哈希、许可证、已知限制和失败恢复证据齐全。
- [ ] 文档不宣称 MVP 已达到生产翻译质量，不把全量 teacher 生成/在线 logits 蒸馏、200M 训练或生产性能混入本 todo。

## 统一 review 与归档

- [ ] TD-01 至 TD-18 全部完成后，对本 todo 和完整 task 集合执行一次统一 review；不为单个 TD 提前创建 review。
- [ ] review 通过后，将 todo、task 集合和 review 记录分别归档到 `work/done/`，并更新 plan/AGENTS 中的状态和相对链接。

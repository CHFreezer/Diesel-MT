# plan: MVP model training

状态：active / todo 已拆分

## 来源

- 项目目标与 MVP 配置：[README](../../README.md)
- 已冻结 tokenizer plan：[mvp tokenizer](mvp-tokenizer.md)
- tokenizer 冻结评审：[mvp tokenizer review](../done/review/mvp-tokenizer.md)
- 已完成部署 plan：[CTranslate2 deployment validation](ctranslate2-deployment.md)
- 部署兼容性评审：[CTranslate2 deployment review](../done/review/ctranslate2-deployment.md)
- teacher 官方模型卡：[tencent/Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B)
- 冻结 teacher artifact：[tencent/Hy-MT2-7B-GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) `Q8_0`
- teacher 许可证：[Hy-MT2-7B LICENSE.txt](https://huggingface.co/tencent/Hy-MT2-7B/blob/main/LICENSE.txt)

## 背景

项目已经冻结五标签、49,152 词表的 `mvp-tokenizer-v0`，并使用随机初始化的微型 M2M100 checkpoint 验证了 Hugging Face 保存/重载、CTranslate2 float32/INT8 转换、完整 token ID 空间、CPU 推理接口和离线部署包。该 tokenizer 只冻结于本 MVP 训练/评测/部署链；现有结果只证明模型接口可部署，不包含平行训练数据、正式训练状态、翻译能力或质量结论。MVP 通过后计划为约 201.5M 正式基线另行训练 65,536 词表并从零训练模型，该计划不属于本 plan 的已实施产物。

下一阶段需要用不可变 tokenizer、经许可/时间/语义审计的真实五语平行语料建立第一个可恢复、可评测、可部署的 student。流程正确性仍由 smoke test 负责；本 MVP 必须让约 60M 模型达到预注册翻译及格线，未过线不能因训练成功结束而宣称完成。

2026-07-16 冻结范围修正：语言名称继续使用 `zho_Hans -> Chinese`、`zho_Hant -> Traditional Chinese`，完整能力从 9 组/18 路扩展为 10 组/20 路。新增两路是简繁中文互转，不增加产品语言、locale 控制或 tokenizer 标签。详细合同见 [`docs/chinese-locale-capability-contract.md`](../../docs/chinese-locale-capability-contract.md)。此前发布的 M0/D0/D1 继续作为不可变 18 路 v1 证据，但不再单独满足完整训练门槛。

2026-07-17 TD-16 第一次边界修正：44,313 条共同 source 上的 1,000-step human-only/distilled A/B 已完成，但它只回答“teacher target 能否替代 human target”，不能被称为已完成 MVP 模型训练。

2026-07-17 长训结论再次修正边界：可配置高吞吐训练器已合入主分支，完整旧 M0 也训练至 step 15,000 并 early-stop；训练稳定但 validation loss 在 step 4,000 后持续回升。进一步审计确认 226,218 条 directed records 只有 11,411 个 semantic/alignment groups，而且 MASSIVE 的 `utt` 是窄领域 locale adaptation，会合法替换地点、媒体、服务和人物。这种数据适合训练系统/路由验证，不足以承担通用机器翻译主体。失效前提最早位于 TD-02；schema v4 已完成 60M ability-first source lock，下一步依次重写并执行 TD-03～TD-05。旧 corpus/checkpoint 均保持不可变诊断证据。

2026-07-17 TD-04 首轮又触发一次来源级质量否决：KFTT 英文侧把日文专名写成罗马字，teacher 向中文生成时无法恢复原始汉字，人工对照原始日文发现 13 条中 10 条存在实体/术语臆译。运行在 14,246 条处停止并保留；英文 source 修订为 UNPC 30k + ALT 5k + 韩英新闻 15k，KFTT 仅保留日文 source 与 human anchor。TD-03 v2 已在新根完成，TD-04 v2 probe 通过后重新开始全量生成。

2026-07-17 TD-04 v2 在 39,130 条处再次触发预注册数量门：`eng→KO` 前 3,130 条仅接收 2,352，24.8% 因长目标截断，按 12k scan 无法达到 10k。隔离探针进一步发现 Hans 长句到 EN/JA/KO/Hant 也受旧固定上限影响。路线门不降低、截断样本不接收；v3 改为 TD-04 自身显式逐路输出上限，并使用 64 slots/32,768 总上下文（512/slot）。真实批次验证无错误、约 13.1 GiB 显存，Hans→KO 截断降至 4/128；v1/v2 journal 均保持不可变失败证据。

2026-07-18 TD-04 v3 完成全部 195,404 次生成和自动 finalize，16 条固定路线均有 10,000 accepted，运行过程无 OOM/网络错误，证明长句上下文与并发修复有效；但固定人工队列在检查 200/756 条时，于 `jpn_Jpan→eng_Latn` 的 20 条 accepted 中发现至少 7 条实质专名、年号或术语错误。问题样本全部追溯到 `kftt-1.0-en-ja` 日文 source，包括把「藤原秀郷／将門」改写成其他人物名、把「安永」写成不存在的 `Eiyo`、把「チューハイ」写成 `tuhao`。这与 v1 的 KFTT 实体臆译属于同类系统性 blocker，只是方向相反；因此 v3 generation 证据保留但不得进入 TD-05，未审记录不得自动 pass。下一身份必须让 KFTT 日英直接消费锁定 human pair，或从 teacher-to-English source 中移除 KFTT 日文，不能再次仅靠自动脚本/截断过滤准入。

2026-07-21 新增近年词汇覆盖要求并重构关键路径：schema v4 保持不可变证据，TD-02 拆为来源清单与 pilot。新来源必须区分数据快照日期与实际内容年代，优先 2020 年以来及持续维护语料；当前链改为 human parallel 硬门、DeepSeek 长上下文整批扫描/稀疏问题 ID 输出、人工校准和 human-first corpus 发布，不再默认由 Hy-MT2 补齐 20 路。

## 语料成熟度口径

语料版本号与成熟度必须分开记录，不能因为一次真实数据运行具有 complete manifest 就把它称为“最小可用训练语料”：

- `fixture`：只服务单元测试和接口契约的微型样本，不得用于真实训练质量结论；
- `smoke`：使用真实来源完成生成、过滤、恢复、回放和发布的端到端冒烟语料，只证明流程正确；
- `mvp`：除覆盖20路和数量门槛外，还必须达到预先冻结的 source-target 语义、实体、数字、否定与脚本忠实度门槛，且按 independent semantic groups 报告能力规模；
- `scale/production`：多来源、全量生成和正式质量验收，仍属于后续 plan。

在本 plan 中，M0 v1 的203,942条human train是不可变的18路历史语料；旧TD-05发布的20路M0共226,218条directed train records。两者继续作为route/system-validation历史证据，但长训后不再满足通用MT `mvp`语料门槛。TD-08的2,263条D0 v1 accepted teacher targets只属于`smoke`。旧D1 composite的44,361条、schema v4 source bank 和 Hy-MT2 v3 accepted 只作为 teacher/训练器诊断证据，不自动继承到新的 human-first corpus。

## 目标

从零初始化并训练 `mvp_e8_d2_v48k` M2M100 语义 Encoder-Decoder 模型：先从版本、许可和内容年代明确的真实平行来源构建 human-first corpus，经确定性硬门、DeepSeek 长上下文批量找错和有界人工校准后发布；再以实际 groups/tokens/路线分布冻结训练配方并达到预注册能力线。随后验证重复训练能力等价、独立验证/一次性正式测试、Hugging Face 离线重载和 CTranslate2 CPU INT8 推理，为计划中的独立 64k tokenizer + 约 201.5M 正式基线提供数据与训练配方证据；本 plan 不实现或训练该正式基线。

本 plan 完成后，项目应能够回答：

- 一条带来源和语言方向的平行样本如何被确定性地切分、清洗、编码并送入模型；
- 五个 tokenizer 语言标签如何映射到四种产品语言、12 个跨语言翻译方向和 2 个中文内部操作；
- 训练能否稳定降低 loss，并从完整 checkpoint 在正确 step 无损恢复训练状态；
- 在不要求模型权重 hash 一致的前提下，重复训练是否能在最短墙钟内达到预注册的总体与20路能力等价门槛；
- 固定测试样例能否由训练后的 Hugging Face checkpoint 和 CTranslate2 模型完成目标语言生成；
- 数据、配置、代码、环境、训练运行、评测结果和部署产物能否通过 manifest 与哈希互相追溯。

## 方向与语言标签口径

产品语言仍为中文、英文、日文、韩文四种。产品层有 12 个有向跨语言翻译方向，并增加 `zho_Hans -> zho_Hant`、`zho_Hant -> zho_Hans` 两个中文内部转换操作。模型层使用冻结 tokenizer 的五个标签：

```text
eng_Latn
zho_Hans
zho_Hant
jpn_Jpan
kor_Hang
```

中文相关的跨语言方向同时容纳 `zho_Hans` 和 `zho_Hant`；非中文方向各对应一个固定标签对。由此先形成 18 个跨语言模型路由：

- 英、日、韩三者之间 6 个有向标签路由；
- 简体中文与英、日、韩之间 6 个有向标签路由；
- 繁体中文与英、日、韩之间 6 个有向标签路由。

再加入 `zho_Hans -> zho_Hant` 和 `zho_Hant -> zho_Hans` 两条中文内部转换路线，形成完整 20 路。这两路不计入 12 个跨语言翻译方向，但属于本阶段训练、评测和部署验收。模型与 teacher 名称继续使用 `Chinese` / `Traditional Chinese`，不新增 locale-specific 控制。两个中文标签直接对齐冻结 FLORES-200 的同名语义：繁体以台湾规范为主要输出基线，港澳正式书面繁体可补充；粤语/广东话是独立语言能力，不论脚本均排除在当前五标签/20 路之外。评测必须同时保留 20 路明细、12 个跨语言方向汇总和 2 个中文内部操作结果。

实际 MVP 数据按 10 组无向 human parallel 关系组织，再在 split 后展开为 20 个有向训练路由。允许低资源关系数量不同，但每组都必须报告独立 semantic groups、pairs、directed records 和 tokens，不能以 English pivot 或无界重复伪装直接路线的数据规模。这里的“五个模型标签”不得简写成“五种产品语言”。

## 固定模型配置

首个正式 student 固定为 `mvp_e8_d2_v48k`：

| 配置项 | 值 |
| --- | ---: |
| `vocab_size` | 49,152 |
| `d_model` | 512 |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 2,048 |
| `encoder_layers` | 8 |
| `decoder_layers` | 2 |
| `encoder_attention_heads` / `decoder_attention_heads` | 8 |
| `tie_word_embeddings` | true |

模型必须从零初始化，不加载或改造 M2M100、NLLB、Hy-MT2 或其他模型权重。词表维度、特殊 token 和语言 token ID 必须来自冻结 tokenizer，不得使用 32k 配置、动态扩词、重排 ID 或原地修改 tokenizer。

训练实现必须探测当前设备/精度/内存并写入 run manifest，但不得按 GPU 型号、固定显存容量或盘符分支。设备内存预算、预留显存、最大利用率、主机与 dataloader 内存预算、micro batch、最大源/目标长度、累积步数、gradient checkpointing 和 worker 数都通过配置控制，并由 TD-14 在真实长度分布上基准冻结。根目录 Git-excluded `LOCAL_HARDWARE.md` 只记录本机事实，不参与语义配置或配置哈希。若锁定后端组合不稳定，可建立经过记录的回退 profile，但不得改变已冻结数据和 tokenizer 身份。

## 范围

包含：

- 定义 human parallel/可选 synthetic 样本 schema、数据源 registry/source lock、许可证与时间记录和确定性 manifest；
- 实现保守清洗、语言标签校验、精确/近重复去重、零截断长度门禁和 train/devtest 隔离；
- 调研并 pilot OpenSubtitles v2024 近年子集、translatewiki、Mozilla、MDN、Wikimedia Content Translation、现有人类锚点及其他明确许可来源；
- 构建按路线、来源、年份和领域分片的 human parallel 预审语料，以 groups/tokens 和实际实收率冻结正式规模，不按计划行数回填低质数据；
- 使用 DeepSeek 对大量带 ID 的句对做长上下文批量扫描，只稀疏输出疑似问题 ID，并以 canary 和未标记抽查校准长上下文漏检；
- 建立覆盖 20 个模型标签路由的微型 fixture，以及小样本过拟合门槛；
- 保留官方 Hy-MT2 7B GGUF Q8_0、D0/D1、v3 和 human/distilled A/B 的冻结诊断证据，但不把它们作为新 corpus 的完成依赖；
- 正式 60M 首轮只消费新 TD-05 human-first manifest；是否需要 synthetic 只由训练后的弱路由 dev 证据触发；
- 从零构建 `mvp_e8_d2_v48k`，实现训练、验证、日志、原子 checkpoint、恢复和最终 Hugging Face checkpoint 发布；
- 首轮未过线时只允许针对已证实弱路由另立一次有界数据补强，来源、人工/synthetic 比例和数量必须在看到补训结果前冻结；
- 对独立验证/测试集报告 loss、生成质量、脚本/目标语言合规性和各方向明细；
- 将训练后 checkpoint 接入已验证的 CTranslate2 float32/INT8 与离线推理路径；
- 为数据边界、编码、collator、checkpoint、恢复、评测和部署回归增加自动化测试。

不包含：

- 未经单独授权的全量 DeepSeek/Hy-MT2 翻译、自动修正或 synthetic 填洞；
- 在线蒸馏、logits/hidden-state 蒸馏或 teacher 权重集成；
- 训练 `e12-d3`、约 200M 目标模型或系统性模型尺寸搜索；
- 生产级 BLEU/COMET 目标、人工翻译质量验收或发布承诺；
- 生产级吞吐、移动 SoC 性能、服务并发或量化调优；
- 原地修改或静默替换 `mvp-tokenizer-v0`；TD-02B 覆盖审计失败时必须阻塞并另立版本化 tokenizer 决策。

## 训练数据契约

每条规范平行样本至少包含：

```text
sample_id
source_id
source_version
license
snapshot_date
content_date_or_year
content_date_status
domain
src_lang
tgt_lang
source_text
target_text
split
```

若样本由 teacher 生成，还必须附加 teacher 模型与权重身份、许可证、prompt/template 版本、解码参数、原始输入 ID 和生成运行 manifest；缺少这些字段的合成样本不得混入正式训练集。DeepSeek 审计另以 overlay 绑定 `batch_id`、输入内容哈希、flag、人工决定和审计配置，不改写规范文本身份。

数据处理遵循以下边界：

- 原始来源、版本、许可证、snapshot date、content date 的值或 `unknown` 状态、文件大小和 SHA-256 必须先锁定，正式构建只消费 source lock；`unknown` 不等于拒绝整个来源，但对应样本不得计入近期内容层；
- `sample_id` 必须由稳定来源身份与规范内容生成，不使用 Python 内置 `hash()`、绝对路径或时间戳；
- 同一平行关系及其反向样本必须按 group 隔离到同一个 split，近重复文本不得跨越 train/dev/test；
- train、dev、test 在生成训练文件前完成隔离，训练运行不得从 test 选择 checkpoint 或调整超参数；
- 清洗不得做英文小写化、简繁转换、日文假名转换或韩文转写；
- 非空、长度、长度比、控制字符、HTML 残留、错误脚本占优等过滤规则必须版本化并报告拒绝原因；
- 方向采样同时记录原始数量和实际抽样权重，低资源方向不能只靠无界重复伪装数据规模；
- 数据构建输出在相同 lock、配置、代码和依赖下必须字节级一致；大体积原始数据和训练文件不提交 Git。

规范 MVP 数据集必须覆盖 5 个标签、10 组无向模型关系和全部 20 个模型路由；实际能力规模用独立 groups、pairs 和 tokens 表示，不以正反展开后的行数替代。TD-02B 预注册首轮 90万～130万个独立 human pairs（正反展开约 180万～260万 directed records），可选扩展为 150万～200万个 pairs；这些是由 5万～10万条 pilot 实收率确认或修订的规划区间，不是降质也要填满的 quota。近期内容层和持续本地化/技术术语层先在 20%～25% 与 5%～10% 区间做实测，最终比例不能以收低质文本达成。繁体必须包含原生 `zho_Hant` 训练样本和独立 dev/test；港澳正式书面繁体可补充，粤语和自动简繁转换不能冒充原生繁体。详细逐关系和领域比例以 [`docs/model-training-dataset-research.md`](../../docs/model-training-dataset-research.md) 为唯一主口径。

## 远程模型与历史蒸馏边界

Hy-MT2 D0/D1、v3 和旧 human/distilled A/B 均已完成各自诊断使命：它们证明运行时可用、训练器能消费 teacher target，同时也证明 Hy-MT2 v3 的实体/术语错误不能只靠自动规则和解码参数修复。所有旧配置、raw/accepted、checkpoint 和报告保持不可变，但不进入新 human-first 数据链。

- 当前 DeepSeek 只执行 TD-04 长上下文批量质量扫描。一次请求包含大量带 ID 的 source-target 句对，响应只列疑似错误 ID、类别和短理由；正常句对不逐条输出 `pass`。
- DeepSeek 未标记记录只表示本轮未发现问题，必须用 canary 召回和未标记人工抽检估计漏检；上下文越大不一定越可靠，批量档位由校准选择。
- DeepSeek 不得自动重译、润色或覆盖 human target；远程响应与规范语料分离，API key 不进入 Git，正式 test 不发送到远程 API。
- TD-04 暂定质量解锁线为 critical canary 召回率不低于 95%、flag 人工有效命中率不低于 70%、未标记严重错误率不高于 1%；pilot 费用 10～30元，首轮累计 150～350元，扩展累计 300～600元，超过600元前重新报告确认，1000元只作悲观硬上限。价格执行前按 DeepSeek 官方页面复核。
- 若 human-first 基线后确需 synthetic，必须另立任务冻结来源、teacher、prompt、数量、质量复审和训练预算；首个候选仅在全局约 5%～10%、单弱路由不超过约 20% 的曝光边界内，与等预算 human-only continuation 比较。student 仍只学习离散 target，不继承 teacher 权重或 tokenizer。

## 编码与训练语义

- tokenizer 只能从 `artifacts/tokenizers/mvp-tokenizer-v0/` 本地加载，并校验冻结 manifest SHA-256 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。
- encoder 输入必须包含正确的 source language token 和 `</s>`；labels 必须包含正确的 target language token 和 `</s>`，padding 位置统一屏蔽为 loss ignore index。
- source/target 截断必须分别统计，报告每个标签路由的样本截断率和 token 长度分布；不能静默丢弃超长尾部而只报告总体均值。
- 每个 batch 的语言方向组成必须可追溯；采样器的随机种子、epoch 和位置必须进入 checkpoint。
- 训练记录至少包含 token 数、optimizer step、学习率、训练/验证 loss、吞吐、显存峰值、wall time、异常跳过数和 checkpoint 身份。
- 必须拒绝非 allowlist 语言、source 与 target 标签相同、空文本、非有限 loss 和词表越界；`zho_Hans <-> zho_Hant` 是新 allowlist 中的合法路线，并执行独立中文内部质量合同。

## Checkpoint 与复现边界

可恢复 checkpoint 至少保存并验证：

- 模型权重、optimizer、scheduler、mixed-precision scaler（若使用）；
- global step、epoch、已消费样本/token 位置、gradient accumulation 相位；
- Python、NumPy、PyTorch CPU/CUDA RNG 状态和采样器状态；
- 模型/训练/数据配置哈希、数据 manifest/lock 哈希、tokenizer 冻结哈希；
- Git commit、工作树状态、Python/依赖/CUDA/GPU 环境和启动命令；
- 完整文件清单、大小、SHA-256 和完成状态。

checkpoint 通过同目录 staging、逐文件校验和最终原子发布生成；不完整、配置不匹配、哈希错误或缺少训练状态的目录必须拒绝恢复。

数据构建和 CPU fixture 继续要求字节级确定性。GPU 训练与中断恢复不要求权重、逐步 loss 或模型 hash 逐 bit 一致；checkpoint 仍必须完整保存 step、学习率、采样位置、优化器和必要 RNG 状态并能安全继续。训练脚本的重复性以预先冻结的 human dev 总体/20路 BLEU、chrF、loss、脚本合规、空输出、source copy、目标控制容差和 time-to-quality 分布验收。输入、tokenizer、配置和 checkpoint 文件 hash 只承担身份/损坏校验，不得为了精确 trace 降低训练热路径吞吐。

## 分阶段门槛

### M0：数据与编码契约

- schema、source lock、split、去重、采样和 manifest 规则固定；
- 微型 fixture 覆盖 20 个标签路由，编码后特殊 token、labels 和 padding mask 全部通过自动化测试；
- 有界真实数据覆盖 5 个标签桶和 10 组无向模型关系，`zho_Hans`、`zho_Hant` 分别具有独立 dev/test；
- train/dev/test 泄漏检查和两次独立构建的字节级复现通过。

### D0（历史）：Hy-MT2 7B 真实数据冒烟语料

- 官方 GGUF Q8_0 teacher artifact、llama.cpp CUDA 后端、许可证、运行环境、语言名称映射、prompt 和 decode profile 全部锁定并可离线重载；
- v1 已在 human dev/reference 上完成 18 路校准；新增两路也已使用现有 `Chinese` / `Traditional Chinese` 名称完成 dev 校准并继续选择 `greedy-v1`；
- D0 v1 的 18 路 smoke 保持不变；新增两路可先执行独立小规模 smoke，raw/accepted/filtered 与完整 provenance 可追溯；
- D0 只验收生成、过滤、人工审查、checkpoint/resume、replay 和 manifest-last 发布，不作为 TD-15/TD-16 的最小可用训练 corpus。

### D1（历史）：Hy-MT2 7B 最小可用蒸馏语料与 20 路 composite

- D1 v1 使用冻结 teacher/profile 完成 18 路生成；40,032 个候选最终接受 39,941 条，每路由 2,211～2,223 条，继续作为不可变跨语言蒸馏证据。
- 新增 `zho_Hans -> zho_Hant`、`zho_Hant -> zho_Hans` 各 2,224 个候选，分别接受 2,213/2,207 条；使用既有语言名称与路线专用过滤，已独立完成人工审查和 4 条精确 replay，未重生成无关路由。
- 已发布引用 D1 v1 与两路 addendum 的 44,361 条 20 路 composite manifest，v1 不变。只有 composite accepted targets 与对应 human references 的交集可进入 TD-15，test 始终隔离。

### M1：训练器与恢复冒烟

- 固定种子从零初始化模型并完成 forward/backward、梯度累积、optimizer/scheduler step 和离线保存/重载；
- 在微型数据上稳定过拟合，loss 明显下降，固定训练样例能够生成预期目标语言和目标文本；
- uninterrupted 与 resumed 短训练满足恢复一致性门槛；故障注入不发布半成品 checkpoint。

### M2：历史诊断、human-first corpus 与 60M 能力训练

- 44,313 条共同 source 的 human-only/distilled 1,000-step A/B 只作为 teacher-target 替代诊断；已选择 human-only step 1,000，并记录纯 distilled 的负结果。
- TD-16A 已将高吞吐实现合入主分支；资源预算、worker、batch、缓存、传输、allocator、optimizer 和日志均由配置/运行时探测决定，训练热路径不追求逐 batch/权重 hash 一致。
- TD-16B 已从零消费完整旧 M0 并在 step 15,000 early-stop；结果只作为训练器/语料诊断，所有 checkpoint 不准入后续阶段。
- schema v4 source bank、Hy-MT2 v3 和 80/20 mixed 方案保持 rejected/blocked 历史证据，不再是当前完成路径。
- TD-02A/TD-02B 先冻结近期来源、许可、实收率和预算；TD-03 构建 human parallel preaudit，TD-04 以 DeepSeek 长上下文稀疏 flag + 人工校准审计，TD-05 发布 human-first corpus；都不得访问 formal test。
- TD-16C 从零训练首个 human-first 60M；TD-16D 只在 dev 暴露明确弱路由后执行或跳过一次有界数据补强，不能预先假定 synthetic 类型或比例；TD-16E 独立复跑验证能力等价、冻结唯一候选。
- TD-16F 才能读取一次正式 test 并发布最终 MVP。无 NaN/Inf、无语言 token/词表越界，20路能力、峰值显存、吞吐和运行 manifest 必须齐全。

### M3：评测与部署回接

- 输出 20 个模型标签路由明细、12 个跨语言产品方向汇总和 2 个中文内部操作结果；至少报告验证/test loss、SacreBLEU、chrF、目标脚本合规率、空输出率和固定样例；
- Hugging Face checkpoint 完全离线重载并通过固定生成回归；
- float32 与 CPU INT8 CTranslate2 转换成功，完整词表与特殊 token 合约继续成立；
- 所有标签路由完成 CT2 target prefix、去 prefix 和 decode 冒烟，并记录量化前后质量差异与 CPU 延迟的诊断值。

## 产物边界

预期的提交内产物包括：

- 模型、训练、数据处理和评测配置；
- 近期来源/许可/时间 registry、未来 64k tokenizer 的 train-side 真人文本候选资格账本、pilot 配置、DeepSeek batch-audit 配置与紧凑审计证据；候选账本不是 64k 训练集或完成身份；历史 teacher lock、运行 profile 和蒸馏 manifest 保持可追溯；
- 数据 registry/source lock schema 与小型测试 fixture；
- 数据准备、训练、恢复、评测和部署接入代码；
- 自动化测试；
- 不含大体积文本或权重的确定性 manifest、汇总指标和统一 review 记录。

大体积原始数据、预处理训练集、optimizer checkpoint、HF 权重和 CTranslate2 转换目录默认是 Git-ignored 的本地运行产物。发布身份由提交内配置/lock、冻结 tokenizer、代码版本和运行产物 manifest 共同确定；不得把可变的 `latest` 目录当作唯一身份。

## 验收标准

- `mvp-tokenizer-v0` 在整个 MVP 阶段无修改，MVP 训练和部署全过程的词表均为 49,152 项且 ID 顺序一致；该约束不延伸到 MVP 通过后计划的新 64k 正式基线。
- 数据来源、许可证、snapshot/content date、硬门、去重、DeepSeek batch/flag、人工校准和输出身份完整可追溯；按 groups/pairs/directed records/tokens 报告 20 路、年份、领域和简繁，所有数据与 FLORES dev/devtest、tokenizer holdout 无泄漏。
- 新来源的真人两侧以稳定 text/document identity 与有向训练展开分离；未来 tokenizer 候选账本只包含 train-side 合格文本并排除 dev/test、holdout、synthetic、canary、quarantine 与路由重复，但不把本阶段描述成已经完成 64k 语料。
- 微型 fixture 覆盖 20 个标签路由，小样本过拟合证明模型实际学习目标映射，而不只是 forward 成功。
- Hy-MT2 D0/D1/v3 与 A/B 诊断证据保持可追溯且不混入新 corpus；DeepSeek 只完成长上下文稀疏找错，canary/真实坏样本召回和未标记抽检满足 TD-02B 预注册门槛，test 未发送到远程 API。
- 原子 checkpoint、错误拒绝和同环境中断恢复通过自动化与运行验收。
- `mvp_e8_d2_v48k` 完成旧共同 source A/B/长训诊断后，以新 TD-05 human-first corpus 达到预注册 20 路能力线，并完成重复能力等价验收；训练成功但能力未过线不算 MVP。
- 正式 test 结果按标签路由与产品方向报告，不把随机模型、训练内样本或合并均值描述为翻译质量结论。
- 训练后的 Hugging Face checkpoint 能离线重载，并成功转换为 CTranslate2 float32 和 CPU INT8；所有 20 个标签路由完成推理接口回归。
- 自动化测试、完整运行命令、版本、哈希、已知限制和失败恢复记录齐全，统一 review 通过后才可归档。

## 风险与停止条件

- 平行数据许可证、来源或可再分发边界不清时，停止将该来源加入训练，不以技术可下载替代许可判断。
- 数据集快照较新但无法证明文本内容较新时，只记录 snapshot recency，不把它计入近期内容层。
- DeepSeek 长上下文 canary/真实坏样本召回或未标记抽检不达标时，减小批次或修改审计协议，不把“模型没返回 ID”当作已通过。
- 训练/test 发生泄漏、反向句对跨 split、teacher 输出缺少 provenance 时，相关数据构建整体无效，必须重建。
- 若未来重启 synthetic，teacher/API 身份、输入范围、费用、输出过滤和人工复审未先冻结时，停止生成，不以旧 D1/v3 直接填入新 corpus。
- 微型数据无法过拟合、恢复路径与连续训练不一致或目标语言 token 行为错误时，不进入真实数据训练。
- 真实训练出现持续 NaN/Inf、显存不足、截断率失控或某方向被采样器饿死时，先修复训练/数据配置，不通过扩大数据或模型掩盖问题。
- 若量化后语言控制失败或质量相对 float32 异常退化，不进入性能优化或发布阶段。

## 后续边界

本 plan 通过必须同时证明“经许可/时间/语义审计的 human parallel corpus -> 约 60M 从零训练 -> 预注册翻译及格线 -> 独立评测 -> CTranslate2 离线推理”。它不代表生产质量，但也不能退化成只跑通流程。以下事项在本阶段统一 review 后另立 plan：

- DeepSeek/Hy-MT2 的全量翻译、自动修正、多轮 teacher 配方搜索和大规模 synthetic 生成；
- 超出 TD-05 human-first corpus 与一次弱路由补强的 production-scale 多来源扩充；
- 64k tokenizer、`e12-d3` 与约 201.5M 正式基线的实现、训练和模型尺寸选择；
- 生产级质量门槛、大规模人工评测、领域评测与生产性能验收。

## 执行拆解

- todo：[MVP model training](../todo/mvp-model-training.md)
- task：[MVP model training](../task/mvp-model-training/index.md)（历史执行到 TD-16B；当前按 TD-02A/TD-02B/新 TD-03～TD-05 重建 human-first 近期语料链；旧 TD-16 suspended）。

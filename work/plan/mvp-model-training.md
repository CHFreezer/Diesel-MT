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

项目已经冻结五标签、49,152 词表的 `mvp-tokenizer-v0`，并使用随机初始化的微型 M2M100 checkpoint 验证了 Hugging Face 保存/重载、CTranslate2 float32/INT8 转换、完整 token ID 空间、CPU 推理接口和离线部署包。现有结果只证明模型接口可部署，不包含平行训练数据、正式训练状态、翻译能力或质量结论。

下一阶段需要用不可变 tokenizer 和真实平行样本建立第一个可恢复、可评测、可部署的 student 训练闭环。该阶段仍是路线 MVP：重点排除数据、标签、loss、checkpoint、生成和部署之间的系统性错误，不以一次训练达到最终产品质量为目标。

2026-07-16 冻结范围修正：语言名称继续使用 `zho_Hans -> Chinese`、`zho_Hant -> Traditional Chinese`，完整能力从 9 组/18 路扩展为 10 组/20 路。新增两路是简繁中文互转，不增加产品语言、locale 控制或 tokenizer 标签。详细合同见 [`docs/chinese-locale-capability-contract.md`](../../docs/chinese-locale-capability-contract.md)。此前发布的 M0/D0/D1 继续作为不可变 18 路 v1 证据，但不再单独满足完整训练门槛。

## 语料成熟度口径

语料版本号与成熟度必须分开记录，不能因为一次真实数据运行具有 complete manifest 就把它称为“最小可用训练语料”：

- `fixture`：只服务单元测试和接口契约的微型样本，不得用于真实训练质量结论；
- `smoke`：使用真实来源完成生成、过滤、恢复、回放和发布的端到端冒烟语料，只证明流程正确；
- `mvp`：覆盖全部 20 路由并达到预先冻结的逐路由数量/质量门槛，允许用于本 plan 的训练与 A/B 结论；
- `scale/production`：多来源、全量生成和正式质量验收，仍属于后续 plan。

在本 plan 中，TD-05 发布的 203,942 条 M0 v1 human train 是 `mvp` 质量级别的跨语言语料，但只覆盖 18 路；TD-05 必须补充第 10 组并发布 20 路 composite 后，完整 M0 才能供 TD-09 验收。TD-08 已发布的 2,263 条 D0 v1 accepted teacher targets 只属于 `smoke`。D1 v1 从 40,032 个候选接受 39,941 条，每个既有路由 2,211～2,223 条，是有效的 18 路跨语言 distilled MVP；TD-08 必须再发布两条中文内部路线的 addendum 与 20 路 composite。v1 产物均不得覆盖、改名或重新解释为 20 路完成。

## 目标

从零初始化并训练 `mvp_e8_d2_v48k` M2M100 语义 Encoder-Decoder 模型，依次完成小样本过拟合、Hy-MT2 7B D0 冒烟与 D1 最小可用离线 sequence-level 蒸馏、human-only 与 distilled 等预算对照、可中断恢复的小规模真实训练、独立验证/测试、Hugging Face 离线重载和 CTranslate2 CPU INT8 推理，形成后续全量蒸馏与更大模型训练可复用的稳定入口。

本 plan 完成后，项目应能够回答：

- 一条带来源和语言方向的平行样本如何被确定性地切分、清洗、编码并送入模型；
- 五个 tokenizer 语言标签如何映射到四种产品语言、12 个跨语言翻译方向和 2 个中文内部操作；
- 训练能否稳定降低 loss，并从完整 checkpoint 在正确 step 无损恢复训练状态；
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

再加入 `zho_Hans -> zho_Hant` 和 `zho_Hant -> zho_Hans` 两条中文内部转换路线，形成完整 20 路。这两路不计入 12 个跨语言翻译方向，但属于本阶段训练、评测和部署验收。模型与 teacher 名称继续使用 `Chinese` / `Traditional Chinese`，不新增 locale-specific 控制。评测必须同时保留 20 路明细、12 个跨语言方向汇总和 2 个中文内部操作结果。

实际 MVP 模型数据按 5 个标签桶组织，必须准备 10 组无向模型关系：英/日/韩之间 3 组，简体中文与英/日/韩 3 组，繁体中文与英/日/韩 3 组，以及 `zho_Hans--zho_Hant` 中文内部转换 1 组。反向扩展后形成 20 个有向训练路由。这里的“5 个标签桶”不得简写为“5 种产品语言”。

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

- 定义平行样本 schema、数据源 registry/source lock、许可证记录和确定性 manifest；
- 实现保守清洗、语言标签校验、精确去重、长度过滤、方向采样和 train/dev/test 隔离；
- 构建覆盖 5 个标签桶、10 组无向模型关系和 20 个有向训练路由的有界 MVP 数据集；
- 建立覆盖 20 个模型标签路由的微型 fixture，以及小样本过拟合门槛；
- 锁定官方 Hy-MT2 7B GGUF Q8_0 teacher 的模型/后端/许可证身份，验证可完全离线运行的 llama.cpp CUDA profile；
- 固定 teacher 语言名称映射、prompt、解码参数和输出过滤；先以 D0 smoke 验证全链，再生成达到逐路由最低规模的 D1 MVP sequence-level 蒸馏训练数据；
- 在相同 student、source 样本和训练预算下比较 human-only baseline 与 distilled 候选；
- 从零构建 `mvp_e8_d2_v48k`，实现训练、验证、日志、原子 checkpoint、恢复和最终 Hugging Face checkpoint 发布；
- 在有界真实平行数据上完成一次小规模 GPU 训练，并保存完整 provenance；
- 对独立验证/测试集报告 loss、生成质量、脚本/目标语言合规性和各方向明细；
- 将训练后 checkpoint 接入已验证的 CTranslate2 float32/INT8 与离线推理路径；
- 为数据边界、编码、collator、checkpoint、恢复、评测和部署回归增加自动化测试。

不包含：

- 全量抓取或批量生成 12 个跨语言产品方向和 2 个中文内部操作的最终训练语料；
- 全量 Hy-MT2 teacher 数据生成、在线蒸馏、logits/hidden-state 蒸馏或 teacher 权重集成；
- 训练 `e12-d3`、约 200M 目标模型或系统性模型尺寸搜索；
- 生产级 BLEU/COMET 目标、人工翻译质量验收或发布承诺；
- 生产级吞吐、移动 SoC 性能、服务并发或量化调优；
- 修改或重新冻结 `mvp-tokenizer-v0`。

## 训练数据契约

每条规范平行样本至少包含：

```text
sample_id
source_id
source_version
license
src_lang
tgt_lang
source_text
target_text
split
```

若样本由 teacher 生成，还必须附加 teacher 模型与权重身份、许可证、prompt/template 版本、解码参数、原始输入 ID 和生成运行 manifest；缺少这些字段的合成样本不得混入正式训练集。

数据处理遵循以下边界：

- 原始来源、版本、许可证、文件大小和 SHA-256 必须先锁定，正式构建只消费 source lock；
- `sample_id` 必须由稳定来源身份与规范内容生成，不使用 Python 内置 `hash()`、绝对路径或时间戳；
- 同一平行关系及其反向样本必须按 group 隔离到同一个 split，近重复文本不得跨越 train/dev/test；
- train、dev、test 在生成训练文件前完成隔离，训练运行不得从 test 选择 checkpoint 或调整超参数；
- 清洗不得做英文小写化、简繁转换、日文假名转换或韩文转写；
- 非空、长度、长度比、控制字符、HTML 残留、错误脚本占优等过滤规则必须版本化并报告拒绝原因；
- 方向采样同时记录原始数量和实际抽样权重，低资源方向不能只靠无界重复伪装数据规模；
- 数据构建输出在相同 lock、配置、代码和依赖下必须字节级一致；大体积原始数据和训练文件不提交 Git。

首轮真实训练数据只要求有界、许可清晰、能够验证训练链路，不要求一次达到最终规模，但规范 MVP 数据集必须覆盖 5 个标签桶、10 组无向模型关系和全部 20 个模型路由。增量构建可以引用不可变 18 路 v1 和新增 addendum，但缺失两条中文内部路线时不得进入 M2 正式训练。繁体必须包含原生 `zho_Hant` 训练样本和独立 dev/test；当前来源 locale 为 `zh-TW`，只作为 provenance 记录。自动转换数据如作为增强使用，必须单独标记生成方式与来源，不能冒充原生繁体或替代原生验收。

## Hy-MT2 7B sequence-level 蒸馏契约

本阶段的“蒸馏”专指离线 sequence-level knowledge distillation：teacher 读取 train source 文本并生成 UTF-8 目标译文，student 仍用普通监督 cross-entropy 学习离散目标 token。teacher 不进入 student forward/backward，不保存或消费 teacher logits、hidden states、attention，也不继承 teacher tokenizer、权重或 decoder-only 架构。

- teacher artifact/backend 冻结为官方 `tencent/Hy-MT2-7B-GGUF` Q8_0、revision `ab8472660ac61fac25f1af43fac2599d52a8a775`、`HY-MT2-7B-Q8_0.gguf` 与 llama.cpp `b10012` CUDA 13.3；规范身份见 `configs/hymt2_teacher_selection.yaml`。TD-06 的量化输出质量基线是相同 revision 的官方原版未量化 BF16，而不是 FP8。18 路 v1 与新增两路都继续使用 `zho_Hans -> Chinese`、`zho_Hant -> Traditional Chinese`；TD-07 只补充两路校准，不改变既有 prompt 名称。
- teacher artifact 必须锁定 Hugging Face revision、完整文件清单/SHA-256、许可证和 llama.cpp/CUDA 后端身份。选定 GGUF 不执行 Hugging Face remote code；正式生成只从本地固定文件启动本地 llama.cpp 服务，不读取浮动 `main` 或运行时下载。已审查的 FP8 Transformers remote-code 路径只保留为未选中基线。
- 本地保留选定 GGUF、llama.cpp 后端及原版 BF16 质量基线，统一放在工作目录下 Git-ignored 的 `artifacts/model-training/runtime/`。该目录只承担模型文件到 RAM/VRAM 的顺序加载与低频只读访问；热 checkpoint、随机写缓存和频繁日志必须使用可配置的受控运行目录，不能写入模型快照目录。具体物理盘映射只记录在根目录 Git-excluded `LOCAL_HARDWARE.md`。
- 官方模型卡给出只输出译文的 prompt 和推荐采样参数，但项目必须在 human dev/reference 小样本上比较确定性解码与官方推荐参数，并在查看大规模输出前冻结唯一 prompt/decode profile。相同输入、artifact 和 profile 必须可重放；若采样导致批次相关非确定性，则不得作为规范蒸馏 profile。
- 正式蒸馏 corpus 只从 train source 生成。仅允许在冻结的有界 human dev 子集上运行 teacher 以校准 prompt/decode，校准输出不得进入 student train；test 不得送入 teacher，不得用于过滤阈值、混合比例或模型选择。
- raw teacher response 与 accepted target 分开保存。过滤至少覆盖空输出、额外解释/prompt echo、目标脚本错误、语言错误、异常长度比、重复、截断和占位符损坏；跨语言路线继续检查 source copy。中文内部两路必须使用独立 source-copy 合同，允许共享汉字、数字、缩写、专名和合法不变短句，并结合人类 reference 与人工审查识别真正未转换输出。
- teacher 输出必须附加模型 revision/hash、prompt version、decode config、seed、输入 sample/group ID、生成运行 manifest、raw/normalized output hash 和过滤版本。Apache-2.0 模型许可证记录不能替代对输入语料许可证与生成数据使用边界的单独审查。
- human-only 与 distilled 训练使用同一 student 初始化规范、source ID 集合、方向采样和有效训练预算。dev/test 始终使用冻结的人类参考，不使用 teacher 译文；是否采用 distilled 候选由预先冻结的 dev 聚合指标、繁简明细和逐路由回退门槛决定。
- D0/D1 v1 的 complete manifest 只对既有 18 路身份成立。D1 v1 保留 39,941 条 accepted；两条新增路线各固定 2,224 个候选、accepted 不少于 2,000，且必须单独通过质量、人工抽检、replay、provenance 和 test 隔离门槛。TD-08 在 20 路 composite 发布前保持 `pending`；TD-15/TD-16 不得消费 D0 或单独的 D1 v1 代替 composite。

本阶段正式蒸馏只允许使用已冻结的官方 GGUF Q8_0 + llama.cpp CUDA 运行路径。官方原版 BF16 只作为量化质量与性能基线，FP8 与 BF16 + bitsandbytes 只作为 TD-06 对比证据；不得在 TD-07/TD-08 中静默回退，也不能用来源不明的社区量化产物替代官方 teacher 身份。若必须改变 artifact、量化或后端，须先形成新的显式选型决策与参考集验收。

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

数据构建和 CPU fixture 要求字节级确定性。GPU 训练不承诺跨 GPU、驱动或 CUDA 版本的权重字节级一致；但同一机器和锁定环境内，中断恢复必须从相同 step、学习率、采样位置和 RNG 状态继续。应以一次短训练对比 uninterrupted 与 resumed 路径，优先要求 loss 与权重一致；若锁定栈存在无法消除的非确定性算子，必须记录算子与环境，并在预先规定的数值容差内验收，不能只凭最终 loss 接近判定恢复正确。

## 分阶段门槛

### M0：数据与编码契约

- schema、source lock、split、去重、采样和 manifest 规则固定；
- 微型 fixture 覆盖 20 个标签路由，编码后特殊 token、labels 和 padding mask 全部通过自动化测试；
- 有界真实数据覆盖 5 个标签桶和 10 组无向模型关系，`zho_Hans`、`zho_Hant` 分别具有独立 dev/test；
- train/dev/test 泄漏检查和两次独立构建的字节级复现通过。

### D0：Hy-MT2 7B 真实数据冒烟语料

- 官方 GGUF Q8_0 teacher artifact、llama.cpp CUDA 后端、许可证、运行环境、语言名称映射、prompt 和 decode profile 全部锁定并可离线重载；
- v1 已在 human dev/reference 上完成 18 路校准；范围修正后还必须使用现有 `Chinese` / `Traditional Chinese` 名称校准两条中文内部路线；
- D0 v1 的 18 路 smoke 保持不变；新增两路可先执行独立小规模 smoke，raw/accepted/filtered 与完整 provenance 可追溯；
- D0 只验收生成、过滤、人工审查、checkpoint/resume、replay 和 manifest-last 发布，不作为 TD-15/TD-16 的最小可用训练 corpus。

### D1：Hy-MT2 7B 最小可用蒸馏语料与 20 路 composite

- D1 v1 使用冻结 teacher/profile 完成 18 路生成；40,032 个候选最终接受 39,941 条，每路由 2,211～2,223 条，继续作为不可变跨语言蒸馏证据。
- 新增 `zho_Hans -> zho_Hant`、`zho_Hant -> zho_Hans` 各 2,224 个候选，每路 accepted 至少 2,000；使用既有语言名称与路线专用过滤，独立完成人工审查和精确 replay，不重生成无关路由。
- 发布引用 D1 v1 与所有 addendum 的 20 路 composite manifest，不覆盖 v1。只有 composite accepted targets 与对应 human references 的交集可进入 TD-15，test 始终隔离。

### M1：训练器与恢复冒烟

- 固定种子从零初始化模型并完成 forward/backward、梯度累积、optimizer/scheduler step 和离线保存/重载；
- 在微型数据上稳定过拟合，loss 明显下降，固定训练样例能够生成预期目标语言和目标文本；
- uninterrupted 与 resumed 短训练满足恢复一致性门槛；故障注入不发布半成品 checkpoint。

### M2：human-only 与 D1 distilled 等预算训练

- 在锁定且覆盖 10 组无向模型关系、20 个有向路由的 composite cohort 上分别完成 human-only baseline 与 Hy-MT2 7B distilled 候选训练；D0 或 D1 v1 单独不得替代 composite；
- 两次运行使用相同 source ID 集合、student 配置/初始化规范、方向采样和有效训练预算，只改变目标/混合配方；
- 无 NaN/Inf、无语言 token/词表越界，训练与验证曲线、峰值显存、吞吐和运行 manifest 齐全；
- 候选选择只依据冻结的 human dev 指标和逐路由回退门槛；test 仅对最终候选执行一次正式评测。若 distilled 未优于 baseline，则记录负结果并停止扩大 teacher 数据，不把它描述为蒸馏成功。

### M3：评测与部署回接

- 输出 20 个模型标签路由明细、12 个跨语言产品方向汇总和 2 个中文内部操作结果；至少报告验证/test loss、SacreBLEU、chrF、目标脚本合规率、空输出率和固定样例；
- Hugging Face checkpoint 完全离线重载并通过固定生成回归；
- float32 与 CPU INT8 CTranslate2 转换成功，完整词表与特殊 token 合约继续成立；
- 所有标签路由完成 CT2 target prefix、去 prefix 和 decode 冒烟，并记录量化前后质量差异与 CPU 延迟的诊断值。

## 产物边界

预期的提交内产物包括：

- 模型、训练、数据处理和评测配置；
- teacher artifact lock、运行 profile、prompt/decode 配置、生成/过滤 manifest 和有界 distilled 数据配方；
- 数据 registry/source lock schema 与小型测试 fixture；
- 数据准备、训练、恢复、评测和部署接入代码；
- 自动化测试；
- 不含大体积文本或权重的确定性 manifest、汇总指标和统一 review 记录。

大体积原始数据、预处理训练集、optimizer checkpoint、HF 权重和 CTranslate2 转换目录默认是 Git-ignored 的本地运行产物。发布身份由提交内配置/lock、冻结 tokenizer、代码版本和运行产物 manifest 共同确定；不得把可变的 `latest` 目录当作唯一身份。

## 验收标准

- `mvp-tokenizer-v0` 在整个阶段无修改，训练和部署全过程的词表均为 49,152 项且 ID 顺序一致。
- 数据来源、许可证、split、过滤、去重、方向采样和输出哈希完整可追溯；数据覆盖 5 个标签桶、10 组无向模型关系和 20 个有向路由，`zho_Hans` 与 `zho_Hant` 均有独立 dev/test，且无 train/dev/test 泄漏。
- 微型 fixture 覆盖 20 个标签路由，小样本过拟合证明模型实际学习目标映射，而不只是 forward 成功。
- Hy-MT2 7B teacher 能从锁定 artifact 完全离线运行；D0/D1 v1 证据保持可追溯，两条中文内部路线各达到至少 2,000 accepted，20 路 composite 的 raw/accepted/filtered 及 provenance 完整，test 从未送入 teacher。
- 原子 checkpoint、错误拒绝和同环境中断恢复通过自动化与运行验收。
- `mvp_e8_d2_v48k` 完成 human-only 与 distilled 等预算 GPU 对照，训练/验证 loss 有限且相对随机初始化基线下降；最终候选选择遵守预先冻结的 dev 与逐路由规则。
- 正式 test 结果按标签路由与产品方向报告，不把随机模型、训练内样本或合并均值描述为翻译质量结论。
- 训练后的 Hugging Face checkpoint 能离线重载，并成功转换为 CTranslate2 float32 和 CPU INT8；所有 20 个标签路由完成推理接口回归。
- 自动化测试、完整运行命令、版本、哈希、已知限制和失败恢复记录齐全，统一 review 通过后才可归档。

## 风险与停止条件

- 平行数据许可证、来源或可再分发边界不清时，停止将该来源加入训练，不以技术可下载替代许可判断。
- 训练/test 发生泄漏、反向句对跨 split、teacher 输出缺少 provenance 时，相关数据构建整体无效，必须重建。
- teacher remote code/revision 未锁定、无法完全离线重载、在显式资源预算内无可接受运行路径或逐路由校准失败时，停止蒸馏数据生成，不退回浮动远端代码或来源不明量化模型。
- teacher 输出的空结果、额外解释、source copy、错语言/错脚本或异常长度在任一路由超过预先冻结门槛时，停止该路由生成并修正 prompt/runtime；不得靠下游 student 训练掩盖 teacher 数据缺陷。
- 微型数据无法过拟合、恢复路径与连续训练不一致或目标语言 token 行为错误时，不进入真实数据训练。
- 真实训练出现持续 NaN/Inf、显存不足、截断率失控或某方向被采样器饿死时，先修复训练/数据配置，不通过扩大数据或模型掩盖问题。
- 若量化后语言控制失败或质量相对 float32 异常退化，不进入性能优化或发布阶段。

## 后续边界

本 plan 通过只证明“有界平行数据 -> 从零训练 -> 独立评测 -> CTranslate2 离线推理”的 MVP 路线成立，不证明模型已达到可用翻译质量。以下事项必须在本阶段统一 review 通过后另立 plan：

- Hy-MT2 7B teacher 的全量蒸馏数据生成、多轮 teacher 配方搜索和大规模质量过滤；
- 更大规模、多来源的 12 个跨语言产品方向与 2 个中文内部操作训练语料构建；
- `e12-d3` 与约 200M 目标配置训练和模型尺寸选择；
- 正式质量门槛、人工评测、领域评测与生产性能验收。

## 执行拆解

- todo：[MVP model training](../todo/mvp-model-training.md)
- task：[MVP model training](../task/mvp-model-training/index.md)（TD-01/TD-02 in_progress；TD-03～TD-05、TD-07/TD-08 pending amendment；TD-06 completed；TD-09～TD-18 pending）。

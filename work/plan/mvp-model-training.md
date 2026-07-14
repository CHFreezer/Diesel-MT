# plan: MVP model training

状态：draft / 待确认

## 来源

- 项目目标与 MVP 配置：[README](../../README.md)
- 已冻结 tokenizer plan：[mvp tokenizer](mvp-tokenizer.md)
- tokenizer 冻结评审：[mvp tokenizer review](../done/review/mvp-tokenizer.md)
- 已完成部署 plan：[CTranslate2 deployment validation](ctranslate2-deployment.md)
- 部署兼容性评审：[CTranslate2 deployment review](../done/review/ctranslate2-deployment.md)

## 背景

项目已经冻结五标签、49,152 词表的 `mvp-tokenizer-v0`，并使用随机初始化的微型 M2M100 checkpoint 验证了 Hugging Face 保存/重载、CTranslate2 float32/INT8 转换、完整 token ID 空间、CPU 推理接口和离线部署包。现有结果只证明模型接口可部署，不包含平行训练数据、正式训练状态、翻译能力或质量结论。

下一阶段需要用不可变 tokenizer 和真实平行样本建立第一个可恢复、可评测、可部署的 student 训练闭环。该阶段仍是路线 MVP：重点排除数据、标签、loss、checkpoint、生成和部署之间的系统性错误，不以一次训练达到最终产品质量为目标。

## 目标

从零初始化并训练 `mvp_e8_d2_v48k` M2M100 语义 Encoder-Decoder 模型，依次完成小样本过拟合、可中断恢复的小规模真实平行数据训练、独立验证/测试、Hugging Face 离线重载和 CTranslate2 CPU INT8 推理，形成后续 Hy-MT2 蒸馏与更大模型训练可复用的稳定入口。

本 plan 完成后，项目应能够回答：

- 一条带来源和语言方向的平行样本如何被确定性地切分、清洗、编码并送入模型；
- 五个 tokenizer 语言标签如何映射到四种产品语言和 12 个有向翻译方向；
- 训练能否稳定降低 loss，并从完整 checkpoint 在正确 step 无损恢复训练状态；
- 固定测试样例能否由训练后的 Hugging Face checkpoint 和 CTranslate2 模型完成目标语言生成；
- 数据、配置、代码、环境、训练运行、评测结果和部署产物能否通过 manifest 与哈希互相追溯。

## 方向与语言标签口径

产品语言仍为中文、英文、日文、韩文四种，产品层保持 12 个有向翻译方向。模型层使用冻结 tokenizer 的五个标签：

```text
eng_Latn
zho_Hans
zho_Hant
jpn_Jpan
kor_Hang
```

中文相关的产品方向同时容纳 `zho_Hans` 和 `zho_Hant` 两种脚本标签；非中文方向各对应一个固定标签对。由此形成 18 个需要覆盖的模型标签路由：

- 英、日、韩三者之间 6 个有向标签路由；
- 简体中文与英、日、韩之间 6 个有向标签路由；
- 繁体中文与英、日、韩之间 6 个有向标签路由。

`zho_Hans -> zho_Hant` 和 `zho_Hant -> zho_Hans` 属于文字转换，不计入本项目翻译方向，也不进入本阶段训练或验收。评测同时保留标签路由明细和 12 个产品方向汇总，不能用简繁合并后的均值掩盖某一脚本缺失或退化。

实际 MVP 模型数据按 5 个语言标签桶组织，必须准备 9 组无向平行语料：英/日/韩之间 3 组，简体中文与英/日/韩之间 3 组，繁体中文与英/日/韩之间 3 组。同一平行关系交换 source/target 后形成 18 个有向训练路由。这里的“5 个标签桶”不得简写为“5 种产品语言”，也不能只准备四个桶后假定繁体会从共享汉字中自然获得翻译能力。

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

在 RTX 4060 Ti 16 GB 本机 profile 上优先使用 BF16、梯度累积和按需 gradient checkpointing。micro batch、最大源/目标长度、累积步数和 dataloader worker 数必须通过显存与吞吐基准确定并写入配置，不在 plan 中凭经验固定。若锁定 CUDA/PyTorch 组合不稳定，可建立经过记录的回退环境，但不得改变已冻结数据和 tokenizer 身份。

## 范围

包含：

- 定义平行样本 schema、数据源 registry/source lock、许可证记录和确定性 manifest；
- 实现保守清洗、语言标签校验、精确去重、长度过滤、方向采样和 train/dev/test 隔离；
- 构建覆盖 5 个标签桶、9 组无向平行语料和 18 个有向训练路由的有界 MVP 数据集；
- 建立覆盖 18 个模型标签路由的微型 fixture，以及小样本过拟合门槛；
- 从零构建 `mvp_e8_d2_v48k`，实现训练、验证、日志、原子 checkpoint、恢复和最终 Hugging Face checkpoint 发布；
- 在有界真实平行数据上完成一次小规模 GPU 训练，并保存完整 provenance；
- 对独立验证/测试集报告 loss、生成质量、脚本/目标语言合规性和各方向明细；
- 将训练后 checkpoint 接入已验证的 CTranslate2 float32/INT8 与离线推理路径；
- 为数据边界、编码、collator、checkpoint、恢复、评测和部署回归增加自动化测试。

不包含：

- 全量抓取或批量生成 12 个产品方向、18 个标签路由的最终训练语料；
- 大规模 Hy-MT2 teacher 推理、在线蒸馏、logits 蒸馏或 teacher 权重集成；
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

首轮真实训练数据只要求有界、许可清晰、能够验证训练链路，不要求一次达到最终规模，但规范 MVP 数据集必须覆盖 5 个标签桶、9 组无向平行语料和全部 18 个模型标签路由。增量构建过程中可以暂时记录缺失路由，但缺失未关闭前不得进入 M2 正式训练。繁体数据可以少于简体，但必须包含原生繁体训练样本和独立 dev/test；自动简繁转换数据如作为增强使用，必须单独标记生成方式与来源，不能冒充原生繁体或替代原生繁体验收。

## 编码与训练语义

- tokenizer 只能从 `artifacts/tokenizers/mvp-tokenizer-v0/` 本地加载，并校验冻结 manifest SHA-256 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。
- encoder 输入必须包含正确的 source language token 和 `</s>`；labels 必须包含正确的 target language token 和 `</s>`，padding 位置统一屏蔽为 loss ignore index。
- source/target 截断必须分别统计，报告每个标签路由的样本截断率和 token 长度分布；不能静默丢弃超长尾部而只报告总体均值。
- 每个 batch 的语言方向组成必须可追溯；采样器的随机种子、epoch 和位置必须进入 checkpoint。
- 训练记录至少包含 token 数、optimizer step、学习率、训练/验证 loss、吞吐、显存峰值、wall time、异常跳过数和 checkpoint 身份。
- 必须拒绝非 allowlist 语言、source 与 target 标签相同、简繁互转路由、空文本、非有限 loss 和词表越界。

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
- 微型 fixture 覆盖 18 个标签路由，编码后特殊 token、labels 和 padding mask 全部通过自动化测试；
- 有界真实数据覆盖 5 个标签桶和 9 组无向平行语料，简体、繁体分别具有独立 dev/test；
- train/dev/test 泄漏检查和两次独立构建的字节级复现通过。

### M1：训练器与恢复冒烟

- 固定种子从零初始化模型并完成 forward/backward、梯度累积、optimizer/scheduler step 和离线保存/重载；
- 在微型数据上稳定过拟合，loss 明显下降，固定训练样例能够生成预期目标语言和目标文本；
- uninterrupted 与 resumed 短训练满足恢复一致性门槛；故障注入不发布半成品 checkpoint。

### M2：有界真实数据训练

- 在锁定且覆盖 9 组无向平行语料、18 个有向路由的小规模真实数据上完成至少一次 `mvp_e8_d2_v48k` GPU 训练；
- 无 NaN/Inf、无语言 token/词表越界，训练与验证曲线、峰值显存、吞吐和运行 manifest 齐全；
- checkpoint 选择只依据 dev，test 仅在候选冻结后执行一次正式评测。

### M3：评测与部署回接

- 输出 18 个模型标签路由明细和 12 个产品方向汇总；至少报告验证/test loss、SacreBLEU、chrF、目标脚本合规率、空输出率和固定样例；
- Hugging Face checkpoint 完全离线重载并通过固定生成回归；
- float32 与 CPU INT8 CTranslate2 转换成功，完整词表与特殊 token 合约继续成立；
- 所有标签路由完成 CT2 target prefix、去 prefix 和 decode 冒烟，并记录量化前后质量差异与 CPU 延迟的诊断值。

## 产物边界

预期的提交内产物包括：

- 模型、训练、数据处理和评测配置；
- 数据 registry/source lock schema 与小型测试 fixture；
- 数据准备、训练、恢复、评测和部署接入代码；
- 自动化测试；
- 不含大体积文本或权重的确定性 manifest、汇总指标和统一 review 记录。

大体积原始数据、预处理训练集、optimizer checkpoint、HF 权重和 CTranslate2 转换目录默认是 Git-ignored 的本地运行产物。发布身份由提交内配置/lock、冻结 tokenizer、代码版本和运行产物 manifest 共同确定；不得把可变的 `latest` 目录当作唯一身份。

## 验收标准

- `mvp-tokenizer-v0` 在整个阶段无修改，训练和部署全过程的词表均为 49,152 项且 ID 顺序一致。
- 数据来源、许可证、split、过滤、去重、方向采样和输出哈希完整可追溯；数据覆盖 5 个标签桶、9 组无向平行语料和 18 个有向路由，简体与繁体均有独立 dev/test，且无 train/dev/test 泄漏。
- 微型 fixture 覆盖 18 个标签路由，小样本过拟合证明模型实际学习目标映射，而不只是 forward 成功。
- 原子 checkpoint、错误拒绝和同环境中断恢复通过自动化与运行验收。
- `mvp_e8_d2_v48k` 完成一次有界真实数据 GPU 训练，训练/验证 loss 有限且相对随机初始化基线下降。
- 正式 test 结果按标签路由与产品方向报告，不把随机模型、训练内样本或合并均值描述为翻译质量结论。
- 训练后的 Hugging Face checkpoint 能离线重载，并成功转换为 CTranslate2 float32 和 CPU INT8；所有 18 个标签路由完成推理接口回归。
- 自动化测试、完整运行命令、版本、哈希、已知限制和失败恢复记录齐全，统一 review 通过后才可归档。

## 风险与停止条件

- 平行数据许可证、来源或可再分发边界不清时，停止将该来源加入训练，不以技术可下载替代许可判断。
- 训练/test 发生泄漏、反向句对跨 split、teacher 输出缺少 provenance 时，相关数据构建整体无效，必须重建。
- 微型数据无法过拟合、恢复路径与连续训练不一致或目标语言 token 行为错误时，不进入真实数据训练。
- 真实训练出现持续 NaN/Inf、显存不足、截断率失控或某方向被采样器饿死时，先修复训练/数据配置，不通过扩大数据或模型掩盖问题。
- 若量化后语言控制失败或质量相对 float32 异常退化，不进入性能优化或发布阶段。

## 后续边界

本 plan 通过只证明“有界平行数据 -> 从零训练 -> 独立评测 -> CTranslate2 离线推理”的 MVP 路线成立，不证明模型已达到可用翻译质量。以下事项必须在本阶段统一 review 通过后另立 plan：

- Hy-MT2 7B teacher 的批量蒸馏数据生成与质量过滤；
- 更大规模、多来源的 12 个产品方向、18 个标签路由训练语料构建；
- `e12-d3` 与约 200M 目标配置训练和模型尺寸选择；
- 正式质量门槛、人工评测、领域评测与生产性能验收。

## 执行拆解

本 plan 确认后再创建对应 todo 和 task；draft 阶段不预先生成空的执行文档。

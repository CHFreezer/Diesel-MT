# plan: mvp tokenizer

状态：done（`mvp-tokenizer-v0` 已于 2026-07-14 冻结）

## 2026-07-14 目标调整与结果

原计划中的“四语 32k/48k 双候选 + CTranslate2 硬前置”已按训练数据指南第 8.5 节收敛为一次性五语冻结目标：

- 语言范围固定为 `eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang`，使用 HPLT 3.0 原生 `cmn_Hant -> zho_Hant`；
- 清洗、五语 exact/MinHash 去重和 train/holdout 隔离升级一次，先 smoke 审计，再生成每语种约 200M 训练字符和约 6M holdout 字符；
- 依据已完成的四语同源比较，只重训 48k（49,152），不重训 32k/64k；旧 32k 仅保留为工程回退和下游吞吐对照；
- 冻结前验收五语覆盖率、tokens/字符、P95/P99、字符丢失、roundtrip、简繁序列差异、保存/重载、language token、微型 M2M100 forward 和 SHA-256；
- CTranslate2 发布验收移为下游部署任务，不再阻塞本次 tokenizer 重训与冻结。

该目标已经完成。默认产物为 `artifacts/tokenizers/mvp-tokenizer-v0/`，冻结根 SHA-256 为 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`；完整记录见 `artifacts/tokenizers/reports/mvp-tokenizer-v0/freeze_acceptance.md`。下文保留最初架构调研和历史候选路线，若与本节冲突，以本节和冻结记录为准。

## 目标

制作 Diesel-MT MVP 阶段可用的中英日韩 tokenizer，用于验证从数据、tokenizer、模型配置、训练、评估到推理的最小闭环。该 tokenizer 必须从零训练，不复用 Meta NLLB-200、M2M100 或其他模型仓库中的 tokenizer 资产。

MVP 阶段优先产出 `32k` 和 `48k` 两个词表规模候选，并通过覆盖率、序列长度膨胀、语言 token 行为和最小训练链路验证选择后续默认版本。

## 范围

本 plan 覆盖 tokenizer 的训练目标、输入要求、特殊 token 约定、导出格式、验证标准和风险边界。

本 plan 不覆盖 tokenizer 训练语料的下载脚本实现，也不覆盖正式翻译模型训练、蒸馏样本生成或生产级 CTranslate2 性能调优；但必须用微型随机 M2M100 模型覆盖 CTranslate2 转换、CPU 加载、目标语言前缀和 decode 的兼容性冒烟，否则 tokenizer 路线不能验收。

## 约束

- tokenizer 必须从项目自有训练语料生成，不能下载、复制或分发 NLLB-200、M2M100、mBART、SMaLL-100 等现成 tokenizer 文件。
- tokenizer 需要支持 `eng_Latn`、`zho_Hans`、`jpn_Jpan`、`kor_Hang` 四个 MVP 必需语言 token；`zho_Hant` 可作为保留 token，但不作为 MVP 覆盖率硬性指标。
- tokenizer 训练不得做英文小写化、中文简繁转换、日文假名转换、韩文罗马化等会破坏原文形态的处理。
- `vocab_size` 必须与后续模型配置中的 embedding 和输出层一致。
- special token ID、语言 token ID、`eos_token_id`、`pad_token_id`、`unk_token_id`、`forced_bos_token_id` 映射必须可序列化、可复现、可测试。
- 技术判断以锁定依赖版本的可执行源码、序列化产物和端到端测试为准，官方文档和 docstring 仅作辅助；源码与文档冲突时不得按文档猜测实现。

## 输入

输入为 tokenizer 专用清洗语料，每种语言应至少提供一个独立文本文件，并附带数据来源、许可证、处理版本、行数、字符数和抽样策略记录。

建议输入目录语义：

```text
data/tokenizer/corpus/mvp/
  eng_Latn.txt
  zho_Hans.txt
  jpn_Jpan.txt
  kor_Hang.txt
  manifest.jsonl
```

语料应按语言均衡抽样，避免英文或中文语料规模过大导致词表被单一语言主导。MVP 允许小规模语料先跑通流程，但必须保留扩大语料规模后可复现重训的接口。

## tokenizer 形态

MVP 采用已核对的 [Transformers 5.13.1 `tokenization_nllb.py`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/nllb/tokenization_nllb.py) 可执行实现定义的 Hugging Face `tokenizers` **BPE + Metaspace** 路线，训练中英日韩共享子词词表。该源码明确声明 `model = BPE`；其中 “Based on Unigram” 的 docstring 与实现冲突，不作为路线依据。5.13.1 是本轮源码调研基线，实际实施仍须在依赖锁定后按相同方法重新核对。

版本路线固定为 Transformers 5.x：训练入口是统一后的 `NllbTokenizer`，它继承 `TokenizersBackend`，本身就是 Rust fast 实现；保存后、CTranslate2 转换前和部署时使用 `AutoTokenizer`，并断言 `tokenizer.is_fast is True`。Transformers 4.x 的 `NllbTokenizerFast` 仅作为历史兼容名称，不得出现在 5.x 主线实现中。若锁定版本的 CTranslate2 冒烟迫使项目退回 4.x，必须作为架构变更重新评审，而不是只替换类名。

规范产物是 `tokenizer.json` 及其 tokenizer/special token 配置。不得将 Unigram 模型改名为 `sentencepiece.bpe.model` 伪装兼容；SentencePiece 二进制模型只能作为经过逐样本等价验证的可选互操作导出。

训练产物必须能够被锁定版本 Transformers 从本地加载，并能服务 M2M100/NLLB 风格的语言 token 控制。`32k` / `48k` 指包含所有语言和特殊 token 的最终 `len(tokenizer)`，其 ID 必须稠密唯一，并与 `M2M100Config.vocab_size`、embedding 和输出层行数完全一致。

改变词表大小和裁剪 NLLB 语言集合不影响 Transformers fast tokenizer 所使用的 Hugging Face Rust `tokenizers` backend：backend 从 `tokenizer.json` 读取 BPE 词表、merge、added/special token 和 post-processor，不要求原始 NLLB 的 256k 词表或 200 种语言。语言集合必须在构造训练种子 tokenizer 时通过 `extra_special_tokens` 显式限制；不得在训练后直接删除 JSON 键、修改 merge 或重排 ID。被裁剪语言必须由 Python 应用层 allowlist 拒绝，不能让 `convert_tokens_to_ids()` 静默退化为 `<unk>`。

必须保留的特殊 token：

```text
<unk>
<s>
</s>
<pad>
eng_Latn
zho_Hans
zho_Hant
jpn_Jpan
kor_Hang
```

默认输入格式遵循 README 约定：

```text
<src_lang> source_text </s>
<tgt_lang> target_text </s>
```

推理时必须能稳定取得目标语言 token 对应的 `forced_bos_token_id`。

## 产物

每个候选 tokenizer 至少产出以下内容：

```text
artifacts/tokenizers/mvp-32k/
artifacts/tokenizers/mvp-48k/
```

每个目录内应包含规范 `tokenizer.json`、可加载 tokenizer 配置、special token 配置、语言 token 到 ID 的映射、训练配置、训练语料 manifest 快照、覆盖率报告和最小编码样例。

CTranslate2 冒烟产物应单独保存转换日志、CPU 运行日志和部署目录说明；部署布局必须保留独立 tokenizer 目录，因为 CTranslate2 `Translator` 消费 token 字符串，不负责执行 Hugging Face tokenizer。

## 验证

验证重点是 tokenizer 是否适合作为 MVP 模型训练入口，而不是单独追求压缩率最优。

必须验证：

- 四种 MVP 语言样例都能 encode/decode，且不会把语言 token 切碎。
- `eng_Latn`、`zho_Hans`、`jpn_Jpan`、`kor_Hang`、`zho_Hant` 的 token ID 稳定存在。
- 对固定中英日韩样例计算 `<unk>` 比例、平均 token 数、字符到 token 比例和极端长样本行为。
- 32k 与 48k 两个候选在中日共享汉字、韩文音节、英文 subword 覆盖上有可比较报告。
- tokenizer 文件能被后续最小训练脚本加载，并能生成与模型配置一致的 `vocab_size`。
- `AutoTokenizer.from_pretrained(..., local_files_only=True)` 返回统一后的 NLLB tokenizer，且 `tokenizer.is_fast is True`；训练前、训练后和保存后重载均不得退回 SentencePiece/Python slow backend。
- 保存后的 `tokenizer.json` 必须满足 `model.type == "BPE"`，BPE/Metaspace 管线和 token ID 在保存、离线重载后保持一致。
- 使用微型随机 `M2M100ForConditionalGeneration` 检查点跑通 `ct2-transformers-converter`，并在 CPU 上用 HF tokenizer 生成 source tokens、通过 `target_prefix=[[tgt_lang]]` 指定目标语言、移除返回的 prefix 后完成 decode。
- 通过 Python `AutoTokenizer` 和 `tokenizers.Tokenizer.from_file()` 分别重载同一份 `tokenizer.json`，逐 ID 对比完整词表，并对固定四语样例比较不添加 special token 时的编码 ID 和 decode 结果。
- 对保留语言断言 `convert_tokens_to_ids()` 返回有效且不等于 `<unk>` 的 ID；对已裁剪语言断言不在 `get_vocab()` 中，并在 Python 应用进入 tokenizer 前由 allowlist 明确报错。

## 验收标准

- 从零训练可重复执行，固定输入 manifest 和随机种子后产物稳定。
- MVP 语言 token 能正确参与 encoder 输入和 decoder 目标语言控制。
- 32k 与 48k 候选均有覆盖率和序列长度报告。
- 至少一套 tokenizer 被标记为 MVP 默认候选，并记录选择理由。
- 产物不包含禁止复用的第三方 tokenizer 资产。
- 锁定的 Transformers 5.x 训练/重载链中 `tokenizer.is_fast is True`，生产代码不引用 `NllbTokenizerFast`。
- CTranslate2 CPU 冒烟通过，且转换词表、模型 embedding/projection 和 tokenizer 最终 ID 空间完全一致。

## 风险

- 语料不均衡会导致词表偏向英文或中文，影响日文、韩文碎片率。
- 简繁中文是否合并会影响中文覆盖和语言 token 设计；MVP 先以 `zho_Hans` 为硬目标，保留 `zho_Hant` 扩展空间。
- 过早固定 32k 或 48k 可能影响后续模型大小；MVP 需要保留重训和对比入口。
- 当前 NLLB BPE 配置为 `byte_fallback=false`、`unk_token="<unk>"`、`fuse_unk=true`：最终基础词表缺少某个 Unicode 字符时，没有字节级兜底，该字符会进入 `<unk>` 路径；同一预分词片段中连续未知字符还可能合并为一个 `<unk>`，因此仅统计 `<unk>` token 比例会低估实际丢失字符数。训练前必须固化 must-cover alphabet，训练后同时报告字符频率加权覆盖率、唯一字符覆盖率和 `<unk>` 对应原文跨度。
- 不得为规避 `<unk>` 直接把 `byte_fallback` 改为 `true`；完整 byte-fallback 还要求 256 个字节 token 和匹配的 decoder，并会改变序列长度与模型行为。如需评估，必须作为独立架构实验与当前源码一致路线对比。
- CTranslate2 Transformers converter 按 `tokenizer.get_vocab()` 的 ID 顺序构建模型词表，且不会替 Transformers tokenizer 自动添加 source special tokens；任何 ID 空洞、总词表维度不一致或 target prefix 错误都会在部署时失败，必须作为本阶段硬验收而不是后续风险备注。
- 自定义词表只与从零初始化且使用同一 ID 空间的模型天然兼容；若改造已有 NLLB checkpoint，单纯 resize embedding 不能重映射 token 语义，必须另行设计权重映射或重新训练。

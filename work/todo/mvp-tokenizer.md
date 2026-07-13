# todo: mvp tokenizer

## 来源

- plan：[mvp tokenizer](../plan/mvp-tokenizer.md)
- README：[Diesel-MT](../../../README.md)（tokenizer 约定、模型配置、MVP 候选参数）
- 上游 todo：[tokenizer dataset fetch script](../done/todo/tokenizer-dataset-fetch-script.md)（已完成，语料就绪）
- 硬件方案：[local-ram-first-hardware.md](../done/task/tokenizer-dataset-fetch-script/local-ram-first-hardware.md)
- 环境：[Python 环境约定](../../../docs/python-environment.md)
- Transformers 5.13.1 源码：[NLLB tokenizer](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/nllb/tokenization_nllb.py)、[`train_new_from_iterator()`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/tokenization_utils_tokenizers.py#L1078-L1261)
- Hugging Face Tokenizers 0.22.2 源码：[BPE 未知字符与 byte fallback 分支](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/models/bpe/model.rs#L382-L462)、[BPE alphabet 构造与裁剪](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/models/bpe/trainer.rs#L273-L323)
- CTranslate2 源码：[Transformers converter](https://github.com/OpenNMT/CTranslate2/blob/master/python/ctranslate2/converters/transformers.py)
- CTranslate2 运行示例：[NLLB](https://opennmt.net/CTranslate2/guides/transformers.html#nllb)

## 目标

从零训练中英日韩 tokenizer，产出 `32k` 和 `48k` 两个候选版本，通过覆盖率、序列长度、语言 token 行为和最小训练链路验证，选定 MVP 默认版本。

本 todo 覆盖 tokenizer 训练、验证、对比、产物输出，以及使用随机初始化微型模型完成 CTranslate2 转换和 CPU 推理冒烟验收。不覆盖正式翻译模型训练、蒸馏样本生成、质量验收或生产级 CTranslate2 性能调优。

## 硬件约束

本阶段不依赖特定硬件型号，但需满足以下最低条件：

| 资源 | 最低要求 | 说明 |
|------|---------|------|
| RAM | 语料大小 × 3 以上 | 四语文本全量加载 + Python 对象开销 + BPE 训练工作集 |
| 磁盘 I/O | 语料存放盘仅做单次顺序读取 | 禁止训练热路径随机 I/O、SQLite 或多流并发读 |
| CPU | 支持 AVX2 的 x86-64 | `tokenizers` Rust backend 的 BPE 训练纯 CPU；线程数不应按超线程盲目设置 |
| GPU | 不需要 | tokenizer 训练阶段纯 CPU；GPU 留给后续 M2M100ForConditionalGeneration 模型训练 |

语料规模参考（`data/tokenizer/corpus/mvp/`）：

| 文件 | 估算字符数 |
|------|-----------|
| `eng_Latn.txt` | ~1.0B |
| `zho_Hans.txt` | ~0.95B |
| `jpn_Jpan.txt` | ~0.96B |
| `kor_Hang.txt` | ~0.80B |

四语均约 1B 字符量级。BPE 32k/48k 的实际训练时间必须由 TD-03 基准测试记录，不在实现前预设。

### RAM-first 执行原则（沿用上游数据获取阶段约定）

1. **语料全量预加载**：训练开始前将四语文本从数据盘一次性顺序读入 RAM；`train_new_from_iterator()` 的迭代器从内存中的文本行 yield，不从磁盘流式重读。
2. **多候选复用**：32k 和 48k 两次训练共用同一份已加载文本，禁止每个候选单独从数据盘重新读取。
3. **产物暂存**：训练完成的 `tokenizer.json` 和配置文件先写到本地高速暂存目录，由后台任务大块搬运到数据盘目标目录，完成校验后原子替换。暂存路径由 `--staging-dir` 参数指定，不硬编码到脚本。
4. **数据盘单路顺序写**：只允许一个后台搬运任务实际写数据盘；manifest 等待全部搬运验证成功后才发布。
5. **禁止数据盘随机 I/O**：训练热路径不创建 SQLite、WAL、临时文件或逐条日志在数据盘上；统计信息在线累计。
6. **CPU 并行度受控**：BPE 训练的 `tokenizers` 并行线程数由 `--num-threads` 参数控制，默认值为 `min(8, os.cpu_count() // 2)`；实现前通过基准测试选择本机吞吐最优值。
7. **内存保护**：训练脚本接受 `--max-memory-gib` 和 `--min-available-memory-gib` 参数；逼近保护线时安全停止（已完成候选继续有效），不回退到磁盘 spill。具体阈值由各机器的实际 RAM 容量决定，不作为项目常量。

## 架构路线（README 已明确，经调研修正）

### 事实源优先级

本路线严格按以下优先级作判断：

1. 项目锁定版本的可执行源码；
2. 实际序列化产物及其可执行行为，例如 `tokenizer.json` 的 `model.type`、token ID 和 encode/decode 结果；
3. 上游自动化测试和本项目端到端测试；
4. 官方文档、docstring 和教程，仅作为辅助说明。

`transformers/main` 只用于调研最新实现；项目版本路线固定为 Transformers 5.x，执行前必须锁定具体 `transformers`、`tokenizers` 和 `ctranslate2` 版本，并以该锁定版本的源码重新核对。若文档与源码冲突，以源码和测试结果为准。

关键发现：已核对的 [Transformers 5.13.1 `tokenization_nllb.py`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/nllb/tokenization_nllb.py) 明确导入 `tokenizers.models.BPE`、声明 `model = BPE`，并使用 `BPE(vocab=..., merges=..., fuse_unk=True, byte_fallback=False)`；预分词器和解码器均为 `Metaspace(replacement="▁", prepend_scheme="always", split=True)`。同一文件 docstring 中的 “Based on Unigram” 与可执行实现冲突，判定为过时或错误描述，不能作为架构依据。5.13.1 是本轮调研基线，TD-01 锁定最终版本后必须重新按源码核对。

### 端到端路线

```text
均衡语料
  -> NllbTokenizer 的 BPE + Metaspace 管线从零训练
  -> tokenizer.json + tokenizer 配置 + 稠密且稳定的 token ID
  -> M2M100ForConditionalGeneration 训练检查点
  -> ct2-transformers-converter（M2M100Loader）
  -> CTranslate2 Translator：HF tokenizer 生成 source tokens，target_prefix 指定目标语言
```

1. **底层分词算法：以源码定义的 BPE 为唯一主线。**
   - 通过锁定版本 `NllbTokenizer.train_new_from_iterator()` 继承同类的 BPE、Metaspace、decoder 和 post-processor 配置；不得另行实现与源码不一致的 Unigram 管线。
   - `train_new_from_iterator(vocab_size=...)` 会移除旧词表和带旧 ID 的 post-processor，以当前 special token 集合重新训练并重建 processor；Rust BPE trainer 先加入 special tokens，再扩展 alphabet 和 merge，因此 `32k` / `48k` 目标包含核心特殊 token、语言 token 和 `<mask>`。
   - 训练前后都要解析 `backend_tokenizer.to_str()` / `tokenizer.json`，断言 `model.type == "BPE"`、`byte_fallback == false`，并核对 Metaspace 参数。
   - 当前源码在没有 `_spm_precompiled_charsmap` 时不额外安装 normalizer；项目不得仅凭文档自行引入 lower、NFKC 或脚本转换。若后续加入 normalization，必须作为独立实验并重新做 CTranslate2 一致性测试。
   - `tokenizer.json` 是 MVP 的规范 tokenizer 产物。不得把 Unigram `.model` 政名为 `sentencepiece.bpe.model`；SentencePiece `.model` 只可作为经过逐样本等价验证的可选互操作导出，不是 CTranslate2 MVP 的前置条件。
   - 改变目标词表大小或裁剪语言集合必须通过重新构造 tokenizer 并重新训练完成；禁止训练后直接编辑 `tokenizer.json` 删除 token、修改 merge 或重排 ID，否则会破坏 added token、post-processor、模型 embedding 和 CTranslate2 词表之间的同一 ID 空间。

2. **上层语言控制：使用锁定版本的 `NllbTokenizer` / `AutoTokenizer`。**
   - [Transformers 4.57.3 `NllbTokenizer`](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/nllb/tokenization_nllb.py) 是 SentencePiece slow 实现，[`NllbTokenizerFast`](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/nllb/tokenization_nllb_fast.py) 才是 Rust fast 实现；Transformers 5.x 已合并两条路径，[5.13.1 `NllbTokenizer(TokenizersBackend)`](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/nllb/tokenization_nllb.py) 本身就是 fast 实现。
   - 项目主线锁定 Transformers 5.x：训练时显式构造 `NllbTokenizer`；保存后、CTranslate2 转换前和部署时通过 `AutoTokenizer.from_pretrained()` 加载，并在每个边界硬断言 `tokenizer.is_fast is True`。
   - `NllbTokenizerFast` 只作为 4.x 历史兼容名称，不得出现在 5.x 主线生产代码；不能继续假设旧版 `NllbTokenizer(vocab_file=...)` 接口。
   - 仅配置本项目语言 token：`eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang`。5.13.1 源码中显式 `extra_special_tokens` 会覆盖默认 `FAIRSEQ_LANGUAGE_CODES`；只有未传入语言集合时才自动加入全部 NLLB 语言。旧版兼容参数是否保留由最终锁定版本决定。
   - Rust backend 不包含“必须使用 200 种语言”或“必须使用原始 NLLB 词表大小”的语义约束；语言码只是 special token。保存后必须验证保留语言均在 `get_vocab()` 中，已裁剪语言均不在其中；Python 应用必须先检查语言 allowlist，不能把未知语言码交给 `convert_tokens_to_ids()` 后静默当成 `<unk>`。
   - 特殊 token 基线固定为 `<s>=0`、`<pad>=1`、`</s>=2`、`<unk>=3`；语言 token 和 `<mask>` 的最终 ID 不靠公式推测，而从保存后重载的 tokenizer 读取并固化为映射文件。
   - 默认非 legacy 行为必须是 source/target 均采用 `<lang> text </s>`。Transformers 生成路径使用 `forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)`。

3. **词表与模型边界：以最终总词表为准。**
   - `32k` / `48k` 指保存并重载后的总词表大小，即 `len(tokenizer)` 和 `max(tokenizer.get_vocab().values()) + 1`，其中已经包含全部特殊 token 和语言 token。
   - 必须验证 token ID 无空洞且唯一：按 ID 排序后严格等于 `0..len(tokenizer)-1`。
   - `M2M100Config.vocab_size`、输入 embedding 行数、输出 projection 行数和最终总词表大小必须完全一致。不能使用可能只表示底层模型词表的 `tokenizer.vocab_size` 代替 `len(tokenizer)`。
   - 本项目模型从零初始化，因此自定义 32k/48k ID 空间可以直接写入模型配置。若未来改为复用已有 NLLB checkpoint，单纯 `resize_token_embeddings()` 只能改变行数，不能把新 token ID 语义正确映射到旧权重，必须作为独立迁移方案处理。

4. **CTranslate2 是硬验收门，而不是后续假设。**
   - CTranslate2 [`M2M100Loader`](https://github.com/OpenNMT/CTranslate2/blob/master/python/ctranslate2/converters/transformers.py#L477-L508) 从 `tokenizer.get_vocab()` 按 ID 排序构造 CT2 词表，并补充 `additional_special_tokens`；因此语言 token 必须在转换前已经存在，且模型词表维度必须匹配。
   - CTranslate2 对 Transformers 模型不会自动添加 source special tokens；source 必须来自同一份 HF tokenizer 的 `encode -> convert_ids_to_tokens` 结果。
   - CTranslate2 推理不用 `forced_bos_token_id` 参数，而是调用 `translate_batch(..., target_prefix=[[tgt_lang]])`。返回 hypothesis 的第一个 token 是该 prefix，解码前按官方 NLLB 示例移除。
   - CTranslate2 转换产物不等同于 tokenizer runtime。部署包必须同时保留独立 HF tokenizer 目录；MVP Python 推理用该目录完成 tokenize/decode，CT2 `Translator` 只消费 token 字符串。

## 已确定决策

- **底层**：锁定版本 `NllbTokenizer` 所定义的 Hugging Face `tokenizers` BPE + Metaspace 管线；不采用 Unigram。
- **规范产物**：`tokenizer.json` + `tokenizer_config.json` + special/language token 映射；不伪造或强制要求 `sentencepiece.bpe.model`。
- **版本与类名**：主线采用锁定的 Transformers 5.x；训练显式使用统一后的 fast `NllbTokenizer`，重载/转换/部署使用 `AutoTokenizer` 并断言 `is_fast is True`。4.x `NllbTokenizerFast` 不进入主线。
- **上层**：`NllbTokenizer` / `AutoTokenizer` 提供 NLLB 风格语言 token 控制；保存后必须由同一锁定版本离线重载。
- **语言裁剪**：训练前通过 `extra_special_tokens` 仅声明五个项目语言 token；Rust fast backend 对裁剪语言集合和 32k/48k 自定义词表兼容，不保留默认 200+ 语言。
- **变更方式**：词表大小或语言集合变化一律重新训练并重新生成全部映射，禁止手工修改 `tokenizer.json` 或沿用旧模型 embedding。
- 模型架构使用 `M2M100ForConditionalGeneration`（M2M100 语义 Encoder-Decoder）。
- 训练语料使用 `data/tokenizer/corpus/mvp/` 下四语文本文件。
- 词表候选为 32k 和 48k；MVP 模型候选配置对应 `e8-d2-v32k`、`e8-d2-v48k`、`e12-d3-v32k`、`e12-d3-v48k`（见 README MVP 配置表）。
- 训练不做英文小写化、中文简繁转换、日文假名转换、韩文罗马化。
- 特殊 token：`<s>`、`<pad>`、`</s>`、`<unk>`、`<mask>`（NLLB 固定）；语言 token：`eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang`。
- 候选 `32k` / `48k` 均指最终 `len(tokenizer)`；必须与下游 `M2M100Config.vocab_size` 和 embedding/projection 行数一致。
- 产物先写到 `--staging-dir` 指定的暂存目录，校验后搬运到目标目录并原子替换；数据盘训练热路径不出现随机 I/O、SQLite 或临时文件。
- CTranslate2 CPU 冒烟为硬验收项：至少跑通微型随机 M2M100 模型的转换、加载、`target_prefix` 推理和 decode；随机模型不做翻译质量判断。
- 不下载、复制或分发 NLLB-200、M2M100 等第三方 tokenizer 资产。
- PyTorch：TD-01 直接装 CUDA 13.2 版（`cu132`），tokenizer 阶段当 CPU 版用；若后续出问题可重装依赖，不作为阻塞风险。
- 训练效率：语料从数据盘顺序读取一次后全量加载到 RAM；训练纯 CPU 不涉及 GPU；多候选训练可复用已加载的文本。

## 待办

### [TD-01 训练环境与依赖]

- [ ] 在 Transformers 5.x 范围内锁定具体 `transformers`、`tokenizers`、`ctranslate2` 和 CUDA 版 `torch`（`cu132`，`--index-url https://download.pytorch.org/whl/cu132`）兼容版本，加入 `requirements.txt` 并生成新的 `requirements.lock`。
- [ ] 记录锁定版本对应的 Transformers `tokenization_nllb.py` 和 CTranslate2 `transformers.py` commit URL；禁止只记录浮动的 `main` / `master` 链接。
- [ ] 用可执行源码断言确认 NLLB `model is BPE`，并确认 CTranslate2 注册了 `M2M100Config -> M2M100Loader`。
- [ ] 在 `.conda` 环境验证 `transformers`、`tokenizers`、`ctranslate2`、`torch` 可正常导入；构造 `NllbTokenizer` 并断言 `is_fast is True`，同时记录 CTranslate2 CPU 支持的 compute types。
- [ ] 生成版本兼容记录：Transformers 版本、实际 tokenizer 类名、基类、`is_fast`、Tokenizers 版本、CTranslate2 转换结果；若 5.x 冒烟失败，停止实施并发起架构变更评审，不得静默切换 4.x。
- [ ] 核对锁定 Transformers 对 `tokenizers` 的版本范围，并用该精确组合完成 `tokenizer.json` 保存、`AutoTokenizer` 重载和底层 `tokenizers.Tokenizer.from_file()` 冒烟。
- [ ] `sentencepiece` 仅在需要验证可选互操作导出时安装，不作为 MVP 规范训练链的默认依赖。

产物：更新后的 `requirements.txt` 和 `requirements.lock`。

### [TD-02 语料输入验证]

- [ ] 逐语言确认 `data/tokenizer/corpus/mvp/` 下四个 `.txt` 文件存在、非空、UTF-8 编码、LF 换行。
- [ ] 核对 `manifest.jsonl` 中的文件 SHA-256 与实际文件一致。
- [ ] 验证语料未经过小写化、简繁转换、假名转换或罗马化（抽样检查）。
- [ ] 统计各语言行数、字符数、UTF-8 字节数，确认四语规模在 1B 字符量级且字符数均衡。
- [ ] 测量从数据盘顺序读取四语文件到内存的耗时，作为训练脚本的 I/O 基线。

产物：语料验收记录。

### [TD-03 NLLB BPE 训练脚本]

- [ ] 实现 tokenizer 训练脚本（如 `scripts/train_tokenizer.py`）。
- [ ] 支持命令行参数：`--corpus-dir`、`--vocab-size`、`--min-frequency`、`--limit-alphabet`、`--output-dir`、`--seed`。
- [ ] 用空的锁定版本 `NllbTokenizer` 调用 `train_new_from_iterator()`，固定继承 BPE + Metaspace 管线；训练脚本不得出现 `model_type='unigram'`。
- [ ] 在训练前和 `train_new_from_iterator()` 返回后均断言 `tokenizer.is_fast is True`，避免意外落入旧版 SentencePiece/Python slow backend。
- [ ] 固定核心特殊 token ID：`<s>=0`、`<pad>=1`、`</s>=2`、`<unk>=3`，并将语言 token 和 `<mask>` 纳入最终 `vocab_size`。
- [ ] 明确 `--vocab-size` 表示包含全部 special/language token 的最终目标大小；训练结束若 `len(tokenizer)` 不是精确 32k/48k，立即失败并检查 special token 数、initial alphabet、`limit_alphabet` 和语料可用 merge 数。
- [ ] 训练前后解析 backend JSON，断言 BPE 类型、`fuse_unk=true`、`byte_fallback=false`、Metaspace 参数和 post-processor 模板符合锁定源码。
- [ ] 在训练配置中固化 must-cover `initial_alphabet`，至少覆盖项目定义的中英日韩基础字符、数字和常用标点；验证 `limit_alphabet >= len(initial_alphabet)`，避免 trainer 在 alphabet 限额下裁掉必保字符。
- [ ] 训练前统计语料的唯一 Unicode 字符、频次及其是否进入 `initial_alphabet`；训练后输出保留/裁剪字符清单，禁止只记录一个 `limit_alphabet` 数字而不审计实际字符。
- [ ] 保持主线 `byte_fallback=false`；不得仅切换该布尔值规避 `<unk>`。若评估 byte fallback，必须另建实验配置，显式加入完整字节 token、匹配 decoder，并重新执行全部覆盖率与序列长度测试。
- [ ] 训练输入按语言均衡采样（不是简单拼接文件），避免某一语言主导词表。
- [ ] 固定采样随机种子和输入批次顺序；记录 `tokenizers` 并行设置，并实测同环境重复训练的字节级或语义级可复现性。
- [ ] 记录训练参数、耗时、最终 `vocab_size` 和语料快照到训练日志。
- [ ] 全量加载四语文本到 RAM 后训练，不从数据盘反复读取。
- [ ] 禁止通过训练后编辑 JSON 的方式裁剪语言或调整词表大小；任何此类变化都必须从受控配置重新训练并生成新 artifact。

产物：可复现的 NLLB BPE 训练脚本。

### [TD-04 NllbTokenizer 构造与语言 token 映射]

- [ ] 按锁定的 Transformers 5.x 源码构造空 `NllbTokenizer`，仅传入五种语言 token；生产代码不得引用 4.x `NllbTokenizerFast` 或旧版 `vocab_file` 路线。
- [ ] 用锁定版本支持的 `extra_special_tokens` 或等价参数替换默认 200+ `FAIRSEQ_LANGUAGE_CODES`，仅保留 `eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang`。
- [ ] 保存后断言 `get_vocab()` 中五个保留语言均存在，抽取若干未保留 NLLB 语言（如 `fra_Latn`、`deu_Latn`、`rus_Cyrl`）断言均不存在；应用层语言校验必须在 `convert_tokens_to_ids()` 之前抛出明确错误。
- [ ] 从保存后重载的 tokenizer 用 `convert_tokens_to_ids()` 生成语言 token -> ID 映射，不依赖当前源码并未提供的 `lang_code_to_id` 属性。
- [ ] 验证 `forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)` 对每个目标语言都可正确获取。
- [ ] 确认 `eos_token_id=2`、`pad_token_id=1`、`unk_token_id=3`，与 M2M100/NLLB 约定一致。
- [ ] 生成并保存语言 token → ID 的 JSON 映射文件。
- [ ] 验证 32k 和 48k 候选中核心 special token ID 一致，并分别记录语言 token / `<mask>` ID。
- [ ] 验证最终总词表 ID 稠密、唯一，且 `len(tokenizer)` 恰好为 32k 或 48k。

产物：可保存和离线重载的 `NllbTokenizer`，以及语言 token 映射 JSON。

### [TD-05 覆盖率与编码质量报告]

- [ ] 对四种语言各准备固定评测样本集（覆盖日常文本、技术文本、混合语言文本和边缘用例）。
- [ ] 统计每个候选 tokenizer 的 `<unk>` token 比例、平均 token 数、字符到 token 膨胀比；同时使用 fast tokenizer 的 offset mapping 统计 `<unk>` 覆盖的原文 Unicode 字符数，避免 `fuse_unk=true` 将连续未知字符合并后低估丢失量。
- [ ] 分别报告字符频率加权覆盖率和唯一字符覆盖率，并列出高频未覆盖字符、按语言/文字系统分类的未覆盖字符以及对应原文样例。
- [ ] 特别关注中日共享汉字的切分一致性、韩文音节覆盖、英文 subword 粒度。
- [ ] 对比 32k 和 48k 在四个语言上的差异，生成可比较表格。
- [ ] 统计各语言极端长句（>500 字符）的 token 数和 `<unk>` 比例。

产物：覆盖率与编码质量报告（`artifacts/tokenizers/reports/coverage-32k.md`、`coverage-48k.md`，及对比摘要）。

### [TD-06 产物保存与 AutoTokenizer 加载验证]

- [ ] 调用 `tokenizer.save_pretrained(artifact_dir)` 保存完整 tokenizer 目录。
- [ ] 确认保存文件至少包含规范 `tokenizer.json`、`tokenizer_config.json` 和必要的 special token 配置；不得生成“文件名是 BPE、内容是 Unigram”的伪兼容文件。
- [ ] 验证 `AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True)` 返回锁定版本统一后的 `NllbTokenizer`，且 `tokenizer.is_fast is True`；不依赖 `facebook/nllb-200-*` 等远端仓库。
- [ ] 解析保存后的 `tokenizer.json`，验证 `model.type == "BPE"`，并对比保存前后完整 `get_vocab()` 和 backend 管线配置。
- [ ] 通过 Python `tokenizers.Tokenizer.from_file(tokenizer_json)` 直接加载规范文件，与 `AutoTokenizer` 比较完整 ID→token 映射以及固定样例在不添加 special token 时的编码和 decode 结果。
- [ ] 验证加载后的 tokenizer 语言 token encode/decode 行为：`eng_Latn` 编码为单一 token，decode 后仍为 `eng_Latn`。
- [ ] 验证 `(<src_lang> source_text </s>, <tgt_lang> target_text </s>)` 格式的 tokenize 结果符合预期。
- [ ] 验证 32k 和 48k 候选中最终 `len(tokenizer)` 与 `M2M100Config.vocab_size` 一致。

产物：可通过 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 加载的完整 tokenizer 目录。

### [TD-07 32k vs 48k 对比与 MVP 默认选定]

- [ ] 汇总两份候选的覆盖率、序列长度、中日韩共享字符表现和子词碎片率。
- [ ] 考虑词表大小对下游模型参数量的影响（32k vs 48k 在 MVP 配置下的 embedding 参数差异约 8M–12M）。
- [ ] 选定 MVP 默认候选并记录选择理由和实施权衡。
- [ ] 若 32k 和 48k 各有优势场景，明确记录各自推荐使用条件。

产物：候选对比报告和 MVP 默认选定记录。

### [TD-08 产物打包与文档]

- [ ] 为 MVP 默认候选生成最终产物目录：
  - 规范 `tokenizer.json`
  - `tokenizer_config.json` 和必要的 special token 配置
  - 语言 token → ID 映射（JSON）
  - 训练配置快照（参数、种子、语料 manifest 引用）
  - 覆盖率报告
  - 最小编码样例（四语 test case）
- [ ] 为非默认候选保留产物，标注为备选。
- [ ] 编写 `artifacts/tokenizers/README.md`，说明目录结构、文件用途和复现步骤。

产物：`artifacts/tokenizers/` 完整目录。

### [TD-09 最小训练链路集成验证]

- [ ] 编写最小验证脚本，用 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 加载 MVP 默认 tokenizer，并立即断言 `tokenizer.is_fast is True`。
- [ ] 验证 encoder 输入构造：`<src_lang> source_text </s>` 的 tokenize 结果正确。
- [ ] 验证 decoder 端：`forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)` 可指定目标语言，`labels` 格式为 `<tgt_lang> target_text </s>`。
- [ ] 用最小 `M2M100Config` 创建一个空模型实例（如 `e8-d2-v32k`），跑通一次 forward pass，确认 tokenizer 输出维度与 embedding 匹配。
- [ ] 在模型初始化前硬断言 `len(tokenizer) == config.vocab_size`；初始化后硬断言 encoder/decoder embedding 和 `lm_head` 行数均等于该值，禁止依赖后续隐式截断、补行或 resize 修复。
- [ ] 验证 encode/decode 往返正确性：四语样例编码后解码不丢失关键内容。
- [ ] 将微型随机模型和同一 tokenizer 保存到一个本地 HF checkpoint 目录，供 TD-10 转换器直接加载。

产物：最小训练链路验证脚本及运行记录。

### [TD-10 CTranslate2 转换与 CPU 推理冒烟]

- [ ] 锁定版本执行 `ct2-transformers-converter --model <local-hf-checkpoint> --output_dir <ct2-dir>`，禁止依赖远端模型或 `trust_remote_code`。
- [ ] 至少转换并加载一个 CPU `int8` 产物；保留 float32 产物作为转换问题的诊断基线。
- [ ] 用同一 HF tokenizer 生成 source token 字符串，确认序列包含 `<src_lang>` 前缀和 `</s>` 后缀；不得依赖 CTranslate2 自动补 special tokens。
- [ ] 对五个目标语言分别执行 `translate_batch(..., target_prefix=[[tgt_lang]], beam_size=1, max_decoding_length=<small>)`，确认不发生 unknown target token、词表越界或模型加载错误。
- [ ] 确认返回 hypothesis 的第一个 token 等于目标语言 prefix，移除该 token 后可由同一 HF tokenizer decode。
- [ ] 检查 CT2 转换词表与 `tokenizer.get_vocab()` 的 ID 顺序完全一致；检查 CT2 配置中的 `bos_token`、`eos_token`、`unk_token` 和 `decoder_start_token`。
- [ ] 将 CT2 模型目录与独立 tokenizer 目录按部署布局打包，验证在新的离线进程中仅从本地路径加载并跑通一次 CPU 推理。

产物：CTranslate2 转换/运行脚本、CPU 冒烟日志和部署目录说明。

### [TD-11 自动化测试]

- [ ] 测试固定输入、采样种子、批次顺序和依赖版本下两次训练产物一致；若 JSON 仅序列化顺序不同，则比较规范化 JSON 和 encode 行为。
- [ ] 测试四种语言样例的 encode/decode 往返正确性。
- [ ] 测试所有语言 token 不会被切分为多个子词（每个语言 token encode 后为单一 ID）。
- [ ] 测试五个保留语言全部存在且 ID 不等于 `<unk>`；测试代表性的已裁剪语言不在词表中，并由 Python allowlist 在 token ID 转换前拒绝。
- [ ] 测试 ID 稠密且 `len(tokenizer)` 与 M2M100Config 的 `vocab_size`、embedding/projection 行数匹配。
- [ ] 测试 special token ID 映射稳定（`<s>=0, <pad>=1, </s>=2, <unk>=3`）。
- [ ] 测试保存前的 `NllbTokenizer` 与离线重载后的 `AutoTokenizer` 对相同文本的 token、ID 和 decode 结果一致。
- [ ] 测试训练前、训练后、离线重载和 CTranslate2 转换器加载边界的 tokenizer 均为 fast backend；项目源码中不允许导入 `NllbTokenizerFast`。
- [ ] 添加边界测试：空字符串、纯空白、纯特殊 token、超长行（>10k 字符）、未知字符/emoji。
- [ ] 增加 alphabet 回归集：罕见汉字与姓名用字、平假名/片假名、Hangul 音节与 Jamo、ASCII/全角数字和标点、常见 emoji；对 must-cover 字符断言不产生 `<unk>`，对非目标字符记录预期行为。
- [ ] 增加连续未知字符测试，验证质量报告按 offset 覆盖的原文字符数计数，而不是把 `fuse_unk=true` 生成的一个 `<unk>` 误计为仅丢失一个字符。
- [ ] 测试 `forced_bos_token_id` 对四个目标语言均可正确获取非零 ID。
- [ ] 添加 CTranslate2 转换与 CPU `target_prefix` 推理冒烟测试；可标记为独立的慢速集成测试，但属于发布前必跑项。

产物：`tests/test_tokenizer.py`。

## 完成条件

- [ ] 从零训练可重复执行，固定输入 manifest、采样种子、输入顺序和锁定依赖后产物稳定。
- [ ] 32k 与 48k 候选均有覆盖率和序列长度报告。
- [ ] 至少一套 tokenizer 被标记为 MVP 默认候选并记录选择理由。
- [ ] 语言 token 作为 extra special tokens 正确参与 `NllbTokenizer` 的 encoder 输入和 Transformers decoder 目标语言控制（`forced_bos_token_id`）。
- [ ] tokenizer 文件可通过 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 加载并能喂入 `M2M100ForConditionalGeneration`。
- [ ] 锁定 Transformers 5.x 后，训练实例和离线重载实例均满足 `tokenizer.is_fast is True`，主线代码不引用 4.x `NllbTokenizerFast`。
- [ ] `tokenizer.json` 明确为 BPE，且训练、保存、重载后的 token ID 和特殊 token 行为一致。
- [ ] 自定义 32k/48k 词表和五语言裁剪在 Transformers fast/Rust backend 上保存重载一致；不存在默认 NLLB 语言残留，也未通过手工 JSON 编辑实现裁剪。
- [ ] 本地 HF checkpoint 可由锁定版本 CTranslate2 转换，并在 CPU `int8` 下通过 source special token、`target_prefix`、hypothesis 去前缀和 decode 的端到端冒烟。
- [ ] 产物不包含禁止复用的第三方 tokenizer 资产（不含 NLLB-200 / M2M100 checkpoint 或 tokenizer 文件）。
- [ ] 测试套件可在无网络条件下通过。
- [ ] 产物目录 `artifacts/tokenizers/` 完整，文档齐备。

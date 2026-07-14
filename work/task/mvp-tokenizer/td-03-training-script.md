# task TD-03: NLLB BPE 训练脚本

状态：done

依赖：TD-01（环境）、TD-02（语料验收）、TD-04（tokenizer 构造规范）

## 目标

实现 `scripts/train_tokenizer.py` 作为一次性训练基线，并实现 `scripts/train_tokenizer_checkpointed.py` 将官方 Rust trainer 的 feed 状态缓存到 D: SSD。从四语均衡语料训练 32k（32,768）和 48k（49,152）两个 BPE tokenizer；训练输入可复核、结果功能等价且可验证。最终候选一旦用于模型训练，必须冻结完整产物并锁定 SHA-256。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- TD-01 锁定的依赖版本和版本兼容记录
- TD-02 语料验收记录（四语文件路径和 I/O 基线）
- TD-04 种子 tokenizer 构造代码

## 执行事项

- 实现 tokenizer 训练脚本 `scripts/train_tokenizer.py`。
- 支持命令行参数：`--corpus-dir`、`--vocab-size`、`--min-frequency`、`--limit-alphabet`、`--num-threads`、`--staging-dir`、`--output-dir`、`--seed`。
- 用 TD-04 定义的构造方式创建空 `NllbTokenizer`，调用 `train_new_from_iterator()`；训练脚本不得出现 `model_type='unigram'`。
- 在训练前和 `train_new_from_iterator()` 返回后均断言 `tokenizer.is_fast is True`，避免意外落入旧版 SentencePiece/Python slow backend。
- 固定核心特殊 token ID：`<s>=0`、`<pad>=1`、`</s>=2`、`<unk>=3`，并将语言 token 和 `<mask>` 纳入最终 `vocab_size`。
- 明确 `--vocab-size` 表示包含全部 special/language token 的最终目标大小；32k/48k 的精确值分别为 32,768/49,152。训练结束若 `len(tokenizer)` 不等于目标值，立即失败并检查 special token 数、initial alphabet、`limit_alphabet` 和语料可用 merge 数。
- 训练前后解析 backend JSON，断言 BPE 类型、`fuse_unk=true`、`byte_fallback=false`、Metaspace 参数和 post-processor 模板符合锁定源码。
- 在训练配置中固化 must-cover `initial_alphabet`，至少覆盖项目定义的中英日韩基础字符、数字和常用标点；验证 `limit_alphabet >= len(initial_alphabet)`。
- 训练前统计语料的唯一 Unicode 字符、频次及其是否进入 `initial_alphabet`；训练后输出保留/裁剪字符清单，禁止只记录一个 `limit_alphabet` 数字而不审计实际字符。
- 保持主线 `byte_fallback=false`；不得仅切换该布尔值规避 `<unk>`。若评估 byte fallback，必须另建实验配置。
- 训练输入按语言均衡采样（不是简单拼接文件），避免某一语言主导词表。
- 固定采样随机种子和输入批次顺序，记录输入顺序 SHA-256 和 `tokenizers` 并行设置。重复训练按功能等价验收，不要求 tokenizer JSON/hash 完全相同；这是因为官方 BPE trainer 在等频 alphabet/merge 边界使用哈希集合和不稳定排序，不承诺跨进程生成相同 ID 排列。
- **RAM-first 一次性基线（保留用于回归对照）**：
  - 训练开始前从数据盘单次顺序读取四语文件，全部文本行加载到 RAM。
  - `train_new_from_iterator()` 的 text iterator 从内存 list 中按均衡采样后的顺序 yield，不从磁盘流式重读。
  - 32k 和 48k 两次训练共用同一份已加载文本，禁止每个候选单独从数据盘重新读取。
  - 训练产物（`tokenizer.json` 等）先写到 `--staging-dir` 指定的暂存目录，校验 SHA-256 后由后台任务大块搬运到 `--output-dir` 目标目录；搬运完成前不发布 manifest。
  - 禁止在数据盘创建 SQLite、WAL、临时文件或逐条日志；统计信息在线累计到内存后在报告阶段一次性写出。
- **大语料首选 checkpointed 执行**：Python 只负责生成一次 canonical snapshot，Rust `feed()` 状态保存到 D:；后续候选训练和中断重跑直接读取 feed checkpoint，不再加载原始语料。
- **CPU 并行度**：支持 `--num-threads` 参数，默认值为 `min(8, os.cpu_count() // 2)`；记录训练吞吐供后续调优。
- **内存保护**：支持 `--max-memory-gib` 和 `--min-available-memory-gib` 参数；逼近保护线时安全停止（已完成候选继续有效），不回退到磁盘 spill。
- 记录训练参数、耗时、最终 `vocab_size`、峰值 RSS、语料快照到训练日志。
- 禁止通过训练后编辑 JSON 的方式裁剪语言或调整词表大小。
- 本任务产出训练脚本，但 32k/48k 的实际训练执行可延后到 TD-05 需要评测对象时再进行。

## 产物

- `scripts/train_tokenizer.py`
- 训练日志模板（含参数、耗时、vocab_size、后端断言结果、字符审计摘要）
- 32k 和 48k 两个候选 tokenizer（正式全量训练后保存到 `artifacts/tokenizers/mvp-32k/` 和 `artifacts/tokenizers/mvp-48k/`；10% 冒烟产物保存在 `artifacts/tokenizers/smoke-10pct/`）

## 验收

- 训练脚本可通过命令行参数指定词表大小、语料目录、输出目录和随机种子。
- 训练前后 `tokenizer.is_fast is True` 断言通过。
- 训练后 `len(tokenizer)` 精确为 32k 或 48k，ID 稠密唯一。
- `backend_tokenizer.to_str()` 中 `model.type == "BPE"`、`fuse_unk=true`、`byte_fallback=false`。
- 同参数同输入重复训练必须满足功能等价：语料及输入顺序一致，normalizer/pre-tokenizer/post-processor/decoder 一致，词表大小与特殊 token 约束一致，must-cover 完整，编码质量和 tokens/char 指标等价；不要求规范 JSON 或文件 SHA-256 相同。
- 最终选定 tokenizer 后冻结完整目录并记录 SHA-256；模型 embedding 与 token ID 绑定，模型训练开始后不得用仅“功能等价”但 ID 重排的 tokenizer 替换。
- must-cover alphabet 中的字符不在训练后被裁剪。
- 代码中不出现 `model_type='unigram'`、`NllbTokenizerFast` 或手工 JSON 编辑。
- 训练输入为语言均衡采样，非简单文件拼接。

## 重写验证记录（2026-07-13）

### 实现与测试

- 删除“seed 配置 + 覆盖 `tokenizer.json`”路线，改用 `NllbTokenizer.train_new_from_iterator()`。
- 四个源文件各顺序读取一次并同时校验 manifest、字节数、行数、字符数和 SHA-256；抽中语料留在 RAM。
- 32k/48k 在同一 worker 内复用同一份内存语料；输入按语言字符数对齐并 round-robin 组成 batch。
- 22,068 字符的 must-cover 集合通过 `initial_alphabet` 直接传入 trainer，并验证 `limit_alphabet >= len(initial_alphabet)`。
- CLI 使用 supervisor/worker 结构：主 supervisor 固定间隔输出 newline heartbeat，不依赖 Rust 进度条、TTY 或回车刷新。
- 产物先写到 staging，保存重载验证后逐文件复制并校验 SHA-256，最后用目录替换发布。
- `tests/test_tokenizer_training.py` 覆盖 seed/allowlist、确定性采样、字符均衡、小 fixture 重复训练、保存重载、artifact manifest 和 supervisor heartbeat。
- checkpoint 实现完成后的全套测试：32 passed；Rust helper `cargo test` 构建通过。

### 10% 真实语料冒烟

四语分别确定性抽取约 10%，再按最小字符数对齐；最终共 1,148,922 行、397,823,714 字符。源文件完整顺序扫描及 SHA-256 校验耗时 56.2 秒，两个候选复用同一份内存语料和完全相同的输入顺序哈希。

| 候选 | 训练耗时 | 最终 vocab | 峰值 RSS | must-cover 缺失 | 产物 |
|------|---------:|-----------:|---------:|----------------:|------|
| mvp-32k | 177.6 s | 32,768 | 18.98 GiB | 0 | `artifacts/tokenizers/smoke-10pct/mvp-32k/` |
| mvp-48k | 265.7 s | 49,152 | 19.38 GiB | 0 | `artifacts/tokenizers/smoke-10pct/mvp-48k/` |

### 冒烟验证结果

- `is_fast is True`：训练前、训练后、staging 重载和最终目录重载均通过 ✅
- `len(tokenizer)` 精确为 32,768 / 49,152 ✅
- `model.type == "BPE"`、`fuse_unk=true`、`byte_fallback=false` ✅
- Metaspace `prepend_scheme="always"` ✅
- 特殊 token ID：`<s>=0, <pad>=1, </s>=2, <unk>=3` ✅
- `<mask>=4`；语言 token ID：`eng_Latn=5, zho_Hans=6, zho_Hant=7, jpn_Jpan=8, kor_Hang=9`（两个候选一致） ✅
- ID 稠密唯一（0..len(tokenizer)-1，无空洞） ✅
- `AutoTokenizer.from_pretrained(local_files_only=True)` 返回 fast `NllbTokenizer` ✅
- 五语言 allowlist 验证通过 ✅
- 22,068 个 must-cover 字符均不产生 `<unk>` ✅
- 两个产物的 6 个 payload 文件均通过 artifact manifest SHA-256 复核 ✅
- supervisor 在 509.6 秒端到端运行中固定每约 5 秒输出一行；最长 Rust 静默阶段仍持续 heartbeat ✅

### review 前保留项

- 本次是 10% 冒烟，不是正式全量候选；正式全量 32,768/49,152 训练仍未执行。
- 小 fixture 可生成完全一致的规范 tokenizer JSON，但这不是大语料验收要求；真实语料重复训练以功能等价为准。
- 10% 训练语料中有 3,645 个非 must-cover 罕见字符未覆盖，需在 TD-05 按频率、文字系统和原文 offset 评估，而不能仅看唯一字符数。
- native merge 阶段本身仍没有中间 checkpoint；越过内存保护线时由 supervisor 终止当前核心训练，但可从已完成的 feed checkpoint 重新开始，不再重复语料加载和 feed。若要保留 merge 的部分进度，仍需 tokenizers/Rust 侧增加可取消且可序列化的训练状态。

## 独立复核记录（2026-07-13）

- `python -m pytest tests/test_tokenizer_training.py -q`：6 passed。
- CLI 参数覆盖任务要求；静态扫描确认 `scripts/` 与 `tests/` 中没有 `NllbTokenizerFast`、`model_type='unigram'` 或伪造 `sentencepiece.bpe.model` 路线。
- 10% 的 32k/48k 两个产物均重新通过离线加载、精确词表大小、fast backend、BPE、`fuse_unk=true`、`byte_fallback=false`、特殊 token、语言 allowlist 和稠密 ID 验证。
- 复核未发现阻断问题，TD-03 标记 done。正式全量 32k/48k 执行按本任务“可延后到 TD-05 需要评测对象时再进行”的约定保留为后续运行项，不阻塞训练脚本本身验收。

## Rust feed checkpoint 与内存复核（2026-07-14）

### checkpoint 流程

- Python 继续复用原有语料入口：manifest 校验、seeded sampling、按字符数对齐和四语 round-robin 顺序均保持不变。
- Python 将最终输入顺序写成 D: SSD 上的 length-prefixed UTF-8 snapshot；Rust helper 通过官方 `Tokenizer::train()` 调用 `BpeTrainer.feed()`，只跳过紧随其后的核心 `train()`。
- feed 完成后序列化 trainer 状态。独立 `--phase train` 只读取 checkpoint，不再扫描原始语料，也不再读取 snapshot；核心训练中断后可以重新从同一 checkpoint 启动。
- `feed_complete` 状态再次执行 prepare 可在约 5 秒内完成 checkpoint 校验并命中缓存。
- 训练日志和 watchdog JSONL 全部写入 `D:\Diesel-MT-tokenizer-stage\`，不得写入 E: HDD。

### 32k 的 5%/10% 实测

| 语料比例 | 训练行数 | 训练字符 | snapshot | feed checkpoint | Rust feed | 核心训练 | 进程树峰值 RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 572,901 | 198,622,416 | 0.426 GiB | 0.305 GiB | 3.36 s | 67.7 s | 10.99 GiB |
| 10% | 1,148,922 | 397,823,714 | 0.854 GiB | 0.596 GiB | 7.70 s | 148.3 s | 18.83 GiB |

两组结果均通过精确 32,768 词表、fast BPE、特殊 token、must-cover 缺失为 0、保存/离线重载和 artifact manifest 校验。10% checkpoint 输入顺序 SHA-256 与原一次性 10% 基线相同。

### 功能等价结论

- 新旧 10% 结果的 normalizer、pre-tokenizer、post-processor 和 decoder 完全一致，8,642 个 merge 的集合完全一致；仅 12 个等频 merge 的排名互换。
- 32,768 个 vocab token 中有 35 个 `limit_alphabet` 边界上的低频字符选择不同。10,000 条真实 snapshot 样本的总 token 数完全相同，仅 4 条的 token 字符串序列不同。
- 同一个 5% checkpoint 重复执行核心训练时也会出现少量低频字符和 ID 重排，证明该现象来自官方 tokenizers 0.22.2 BPE trainer 的等频排序行为，不是 checkpoint 输入丢失或损坏。
- 因此 checkpoint 路线通过功能等价验收；不追求 tokenizer JSON/hash 相等。最终正式 tokenizer 仍须在模型训练前锁定完整产物及 SHA-256。

### 全量内存结论

- 旧的一次性 32k 全量尝试在核心阶段仍持续增长时达到进程树 RSS 80.092 GiB，由 watchdog 按 80 GiB 上限终止，没有发布半成品。
- 用 5%/10% 峰值分别做幂律、带基础开销幂律和线性外推，全量核心训练的进程树峰值约为 **113–160 GiB**，中间估计约 **137 GiB**。
- 全量 snapshot 预计约 8.54 GiB，feed checkpoint 预计约 5.84 GiB；磁盘和 feed 阶段不是瓶颈，瓶颈是官方 BPE 的 tokenize words/count pairs/merge 内存结构。
- 当前机器约 100 GiB 可用物理内存，不应直接运行全量 32k。在 80 GiB 进程上限下，可承受比例估算约 49%–64%；为保留安全余量，本机最多按 **50%** 语料训练。正式全量建议使用至少 **192 GB RAM** 的机器。

## 50% 语料训练记录（2026-07-14）

按本机上限使用 checkpointed 路线顺序训练 32k/48k 两个候选，命令参数为 `sample_fraction=0.5`、`seed=20260713`、16 线程、80 GiB 进程树 RSS 上限、系统可用内存 16 GiB 告警/4 GiB 终止。checkpoint、staging、控制台日志和逐秒 watchdog 日志全部位于 `D:\Diesel-MT-tokenizer-stage\checkpoints\mvp-tokenizer-50pct-20260714-v1\`。

- 四语源文件完成顺序扫描和 manifest/SHA-256 校验；抽样并按字符数对齐后共有 5,763,471 行、1,998,439,416 字符、23,635 个唯一字符。
- canonical snapshot 为 4,604,567,336 B（SHA-256 `23d21940dac41a5cb9bfc40232d1410375a677b1920503d6051960c432f7482b`）；feed checkpoint 为 3,051,402,382 B（SHA-256 `6aad79b997d3b001b53f95ff2643ab99f147f81ca83d21190a24a26e4cb4c6fe`）。
- 输入顺序 SHA-256 为 `f276da8d4f0da7cb64f78efd42830c28b95a9ba37a5481cc1c561412dc999698`；state config fingerprint 为 `a1c0fcca1bc597ba185382ebaa7d9d36aa1bbc24ad4fd3fc779c55fc3235face`。
- 语料加载/均衡耗时 162.6 秒，Rust feed 耗时 466.5 秒；端到端运行 5,763.2 秒。

| 候选 | Rust 核心训练 | 含验证/发布耗时 | 进程树峰值 RSS | 系统最低可用内存 | 最终 vocab | must-cover 缺失 | 产物 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mvp-32k | 2,359.1 s | 2,399.1 s | 78.977 GiB | 22.529 GiB | 32,768 | 0 | `artifacts/tokenizers/50pct/mvp-32k/` |
| mvp-48k | 2,593.6 s | 2,626.9 s | 75.740 GiB | 25.345 GiB | 49,152 | 0 | `artifacts/tokenizers/50pct/mvp-48k/` |

独立复核确认两个候选均可由 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 加载为 fast `NllbTokenizer`；backend 为 BPE，`fuse_unk=true`、`byte_fallback=false`，ID 稠密，特殊 token/语言 token ID 与 10% 冒烟一致。各自 6 个 payload 文件均通过 artifact manifest 的字节数和 SHA-256 复核。`tokenizer.json` SHA-256 分别为：

- 32k：`ad750d7a68cb4e3f1ce7b7781152826e59f56f9fdf8710e8ca84c7dfc4e17684`
- 48k：`9ad394a5a10fd9575f3c3cdcd3b3b8d87975abb67fda779ac795e9a375c7653d`

本次复核发现 checkpointed 产物最初只在 state manifest 记录 `sample_fraction`/`seed`，候选 `training_meta.json` 缺少这两个顶层字段，可能导致后续评测误判为全量。训练脚本已改为从不可变 checkpoint state 写入 `seed`、`sample_fraction`、采样/均衡算法、state fingerprint 和 snapshot 摘要；本次两个候选的元数据与 artifact manifest 已原子更新，tokenizer JSON 和 token ID 未修改。

最终回归：`.conda\python.exe -m pytest -q` 为 33 passed；独立 `--phase prepare` 复核在 14.9 秒内命中并验证既有 feed checkpoint。

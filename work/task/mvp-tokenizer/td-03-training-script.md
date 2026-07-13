# task TD-03: NLLB BPE 训练脚本

状态：review

依赖：TD-01（环境）、TD-02（语料验收）、TD-04（tokenizer 构造规范）

## 目标

实现 `scripts/train_tokenizer.py`，用锁定版本 `NllbTokenizer.train_new_from_iterator()` 从四语均衡语料训练 32k（32,768）和 48k（49,152）两个 BPE tokenizer，产物可复现、可验证。

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
- 固定采样随机种子和输入批次顺序；记录 `tokenizers` 并行设置，并实测同环境重复训练的字节级可复现性。
- **RAM-first 执行**：
  - 训练开始前从数据盘单次顺序读取四语文件，全部文本行加载到 RAM。
  - `train_new_from_iterator()` 的 text iterator 从内存 list 中按均衡采样后的顺序 yield，不从磁盘流式重读。
  - 32k 和 48k 两次训练共用同一份已加载文本，禁止每个候选单独从数据盘重新读取。
  - 训练产物（`tokenizer.json` 等）先写到 `--staging-dir` 指定的暂存目录，校验 SHA-256 后由后台任务大块搬运到 `--output-dir` 目标目录；搬运完成前不发布 manifest。
  - 禁止在数据盘创建 SQLite、WAL、临时文件或逐条日志；统计信息在线累计到内存后在报告阶段一次性写出。
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
- 同参数同输入两次训练产物规范 JSON 一致，encode 行为一致。
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
- `tests/test_tokenizer_training.py` 覆盖 seed/allowlist、确定性采样、字符均衡、两次训练 tokenizer JSON 一致、保存重载、artifact manifest 和 supervisor heartbeat。
- 全套测试：24 passed。

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
- 小 fixture 已验证同输入两次训练的规范 tokenizer JSON 完全一致；10% 或全量语料尚未做第二次昂贵重训复核。
- 10% 训练语料中有 3,645 个非 must-cover 罕见字符未覆盖，需在 TD-05 按频率、文字系统和原文 offset 评估，而不能仅看唯一字符数。
- native merge 阶段无法安全中断；内存保护在加载和 iterator feed 阶段检查，merge 阶段由 supervisor 监控 RSS。若要在 merge 阶段越线时保留当前进程，需要 tokenizers/Rust 侧增加可取消训练接口。

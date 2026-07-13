# task TD-02: 语料输入验证

状态：pending

依赖：无

## 目标

确认 `data/tokenizer/corpus/mvp/` 下四个语言文本文件和 manifest 的完整性、编码一致性和规模均衡性，为训练脚本提供可信的输入基线。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- `data/tokenizer/corpus/mvp/eng_Latn.txt`
- `data/tokenizer/corpus/mvp/zho_Hans.txt`
- `data/tokenizer/corpus/mvp/jpn_Jpan.txt`
- `data/tokenizer/corpus/mvp/kor_Hang.txt`
- `data/tokenizer/corpus/mvp/manifest.jsonl`

## 执行事项

- 逐语言确认四个 `.txt` 文件存在、非空、UTF-8 编码、LF 换行。
- 核对 `manifest.jsonl` 中的文件 SHA-256 与实际文件一致。
- 验证语料未经过小写化、简繁转换、假名转换或罗马化（按语言分别抽样检查）。
- 统计各语言行数、Unicode 字符数、UTF-8 字节数，确认四语规模在 1B 字符量级且字符数均衡（差异不超过 25%）。
- 测量从数据盘**单次顺序读取**四语全部文件到内存的耗时和吞吐，作为训练脚本的 I/O 基线。记录单文件读取速度和四语串行全量加载总时间，确认读取模式为顺序而非随机。
- 按语言抽样检查文本行内容：确认英文含大小写、中文含简繁汉字、日文含假名与汉字、韩文含 Hangul 音节，不存在被清洗过度的迹象。
- 记录各语言的文件大小、行数、字符数、SHA-256、读取耗时和抽样检查结论。

## 产物

- 语料验收记录（Markdown 或 JSON），包含上述全部统计数据。

## 验收

- 四个文件均 UTF-8 编码、LF 换行、非空。
- 文件 SHA-256 与 manifest 一致。
- 四语字符数均在 0.8B–1.2B 范围内。
- 抽样检查未发现小写化、简繁转换、假名转换或罗马化。
- I/O 基线数据可用于 TD-03 训练脚本的性能评估。

## 验证记录

（待填写）

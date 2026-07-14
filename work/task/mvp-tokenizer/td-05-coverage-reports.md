# task TD-05: 覆盖率与编码质量报告

状态：in_progress

依赖：TD-03（训练脚本与候选 tokenizer）

## 目标

对 32k 和 48k 两个候选 tokenizer 生成覆盖率、序列长度和子词质量报告，特别关注 BPE `fuse_unk=true` 下的实际字符丢失量、中日共享汉字和韩文音节覆盖。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- TD-03 训练产出的 32k 和 48k tokenizer
- 四语评测样本集（本任务内准备）

## 执行事项

- 对四种语言各准备固定评测样本集，覆盖：日常文本、技术/新闻文本、混合语言文本、纯数字/标点、边缘用例（空行、极短行、超长行）。
- 评测样本应从语料中随机抽取，每种语言至少 500 条，并排除训练时可能见过的高频重复行。固定随机种子确保可复现。
- 统计每个候选 tokenizer 的 `<unk>` token 比例、平均 token 数、字符到 token 膨胀比。
- 使用 fast tokenizer 的 offset mapping 统计 `<unk>` 覆盖的原文 Unicode 字符数，避免 `fuse_unk=true` 将连续未知字符合并为一个 `<unk>` 后低估丢失量。
- 分别报告字符频率加权覆盖率和唯一字符覆盖率，并列出高频未覆盖字符、按语言/文字系统分类的未覆盖字符以及对应原文样例。
- 特别关注中日共享汉字的切分一致性、韩文音节覆盖、英文 subword 粒度。
- 对比 32k 和 48k 在四个语言上的差异，生成可比较表格。
- 统计各语言极端长句（>500 字符）的 token 数和 `<unk>` 比例。
- 报告不要求 48k 在所有指标上优于 32k；如实记录差异即可。

## 产物

- `artifacts/tokenizers/reports/coverage-32k.md`
- `artifacts/tokenizers/reports/coverage-48k.md`
- 对比摘要（`artifacts/tokenizers/reports/comparison.md`）
- 评测样本集（JSONL，按语言分文件）

## 验收

- 两个候选均完成四语评测。
- `<unk>` 统计区分 token 比例和原文字符比例。
- 字符覆盖率报告区分频率加权和唯一字符两个维度，并列出高频未覆盖字符和样例。
- 中日共享汉字、韩文音节和英文 subword 有专项分析。
- 32k 与 48k 对比表格包含所有指标。
- 评测样本集已固定，可供 TD-11 回归测试使用。

## 验证记录

### 实现与测试（2026-07-13）

- 新增 `scripts/evaluate_tokenizers.py`：
  - 按 SHA-256 priority 固定种子抽样，每种语言 500 条去重语料样本，并保证至少 25 条超过 500 字符；
  - 另加每语种 8 条独立压力样例，覆盖日常、技术/新闻、混合语言、数字/标点、空行、极短行和罕见 Unicode；压力样例不进入主覆盖率汇总；
  - 用 fast tokenizer offset mapping 将融合 `<unk>` 映射回全部原文 Unicode 字符；
  - 输出 `<unk>` token 比例、原文字符丢失率、频率加权/唯一字符覆盖率、平均长度、P50/P95/P99/Max、tokens/char、精确 roundtrip 和词表利用率；
  - 输出中日共享汉字、韩文音节/Jamo、英文 word fertility 专项指标，以及高频未覆盖字符、文字系统分类和原文上下文；
  - 保存机器可读 JSON 和 Markdown 候选/对比报告。
- 新增 `tests/test_tokenizer_evaluation.py`，覆盖固定抽样可复现性、长句配额、主样本与压力样例隔离，以及 `fuse_unk=true` 连续未知字符按 offset 计数。测试结果：3 passed。

### 固定评测集

- 本地路径：`data/tokenizer/evaluation/mvp/`。
- 每语种 500 条语料样本 + 8 条压力样例，共 2,032 条。
- 抽样种子：`20260713`；语料 manifest SHA-256：`11daef6a8b38c7dc66dc17d51ab4ab2ab9053a0816340261ad76e53740be0477`。
- 四个源文件均在抽样时完成全文件 SHA-256、行数和字符数复核；评测集 manifest 可在后续正式评测时直接复用，无需重新扫描 9.1 GB 语料。

### 10% 冒烟报告

- 路径：`artifacts/tokenizers/reports/smoke-10pct/`。
- 32k：tokens/char `0.5821`，`<unk>` token 比例 `0.002921%`，原文字符丢失率 `0.001700%`，频率加权字符覆盖率 `99.998300%`。
- 48k：tokens/char `0.4955`，`<unk>` token 比例 `0.003718%`，原文字符丢失率 `0.001842%`，频率加权字符覆盖率 `99.998158%`。
- 48k 相比 32k 将总体 tokens/char 降低约 14.9%，但 10% 冒烟下罕见字符覆盖没有稳定提升；该结果仅验证评测链路，不用于 TD-07 最终选型。
- 两候选的 offset 统计均无未映射 `<unk>`；共享中日汉字 1,100 个均覆盖且 source-language standalone split mismatch 为 0；样本内韩文音节/Jamo 频率和唯一覆盖率均为 100%。

### 50% 四语同源对比（2026-07-14）

- 路径：`artifacts/tokenizers/reports/50pct/`；32k/48k 均使用同一 checkpoint provenance、`sample_fraction=0.5`、1,998,439,416 个训练字符。
- 32k：tokens/char `0.5820`，P95/P99 `610/1550`，原文字符丢失率 `0.001275%`，频率加权覆盖率 `99.998725%`。
- 48k：tokens/char `0.4953`，P95/P99 `519/1303`，原文字符丢失率 `0.001275%`，频率加权覆盖率 `99.998725%`。
- 48k 将总体 tokens/char 降低 `14.89%`，P95/P99 分别缩短 `14.92%`/`15.94%`；英语、简中、日语、韩语分别缩短 `16.20%`、`12.33%`、`15.61%`、`15.80%`。
- 两者均有 8 个 `<unk>` token、对应 9 个丢失原文字符，精确 roundtrip 均为 `99.65%`；48k 显示的 `<unk>` token 比例略高仅因为总 token 数更少，不是覆盖退化。
- 48k 英文 pieces/word 从 `1.7539` 降到 `1.4602`，单 piece 单词比例从 `59.85%` 提升到 `72.25%`。
- 两候选均通过离线重载和内存中的微型 M2M100 forward；相关评测/训练测试为 `11 passed`。
- 本轮据此将 48k 选为五语一次性重训的默认规模；这是四语同源选型，不替代新增 `zho_Hant` 后的五语最终验收。

### 待完成

- 将训练和固定评测集扩展到原生 `zho_Hant`，并在收紧清洗后的五语语料上重训 48k。
- 对五语候选复用同一指标体系生成最终报告，完成本任务验收并将状态更新为 done。

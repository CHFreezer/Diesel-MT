# task TD-05: 覆盖率与编码质量报告

状态：pending

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

（待填写）

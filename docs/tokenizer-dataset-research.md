# 中英日韩 tokenizer 数据集调研

## 调研范围

调研日期：2026-07-13。

本调研面向 Diesel-MT MVP 阶段的中英日韩共享 SentencePiece tokenizer，目标语言为 `eng_Latn`、`zho_Hans`、`jpn_Jpan` 和 `kor_Hang`。排序重点考察文本质量、四语覆盖、获取便利性、语言及文字系统标注、许可证可审计性和小规模抽样能力。

tokenizer 训练只需要覆盖目标语言分布，不要求使用句对齐的平行语料。因此，MVP 优先使用质量可控的单语语料，平行语料留给翻译模型训练阶段。

## 综合排序

| 排名 | 数据集 | 适用结论 | 主要限制 |
| --- | --- | --- | --- |
| 1 | [HPLT 3.0](https://huggingface.co/datasets/HPLT/HPLT3.0) | 最适合作为主语料。覆盖 `eng_Latn`、`cmn_Hans`、`jpn_Jpan`、`kor_Hang`，按 WDS 质量分数排序，并支持流式读取 | 英文和中文未全局去重，需要在项目侧补充去重 |
| 2 | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 与 [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 清洗质量和下载体验好，适合作为备用广域网页语料 | 英文需要单独使用 FineWeb；FineWeb2 中文为 `cmn_Hani`，不能直接区分简繁 |
| 3 | [Wikimedia Dumps](https://dumps.wikimedia.org/backup-index.html) | 文本规范、来源稳定，适合补充百科文本、专名和长尾字符 | 领域偏百科；中文源文本包含简繁差异；需要遵守 Wikimedia 文本许可证 |
| 4 | [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) | 深度清洗和近似去重，四种语言规模充足 | 需要登录并接受 gated 条件；数据约 16TB；中文不分简繁；许可证继承 mC4 和 OSCAR |
| 5 | [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) | 可按语言加载并带有人工审计，可作为备用来源 | 官方审计指出中文噪声较多且混有繁体，日文混有少量英文；日韩清洗受空格启发式限制 |
| 6 | [mC4](https://www.tensorflow.org/datasets/catalog/c4) | 语言覆盖全面，工具资料成熟 | 体量约 38.49TiB，噪声和重复较多，已存在质量更高且更易抽样的替代品 |

## 推荐方案

MVP 默认启用 HPLT 3.0，Wikimedia 作为高质量补充，FineWeb 和 FineWeb2 作为可选备用源。

建议的数据源映射：

| 项目语言 | HPLT 3.0 来源 | 项目输出 |
| --- | --- | --- |
| 英文 | `eng_Latn` | `eng_Latn.txt` |
| 简体中文 | `cmn_Hans` | `zho_Hans.txt` |
| 日文 | `jpn_Jpan` | `jpn_Jpan.txt` |
| 韩文 | `kor_Hang` | `kor_Hang.txt` |

建议的 MVP 抽样策略：

- 优先抽取 HPLT 3.0 的高质量 WDS 分片，例如 10 至 8 分区。
- 英文、日文和韩文可按约 80% HPLT 3.0、20% Wikimedia 混合。
- 简体中文默认只使用 `cmn_Hans`；除非能在不做简繁转换的前提下验证文本，否则不默认混入中文 Wikimedia。
- 每种语言先准备约 10 亿至 20 亿个清洗后 Unicode 字符，并按字符量而不是文档数做语言均衡。
- 对英文和中文额外执行精确去重和 MinHash 近似去重。
- 固定数据版本、抽样种子和字符预算，并在 manifest 中记录每个来源的实际占比。

## 暂不采用

[OSCAR 23.01](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) 当前显示暂停访问，不满足 MVP 的易获取要求。其 CC0 声明也只覆盖数据包装、元数据和标注，不覆盖网页原文版权。

FLORES 等翻译评测集不应加入 tokenizer 训练语料，以避免污染后续评测。

## 许可证与审计要求

HPLT 的 CC0、FineWeb 的 ODC-By 等数据集许可证主要描述数据包装、元数据或汇编权利，并不自动消除网页原文的版权。获取脚本和 manifest 至少需要保留：

- 数据集名称、版本或快照日期和官方主页。
- 数据集许可证及上游条款链接。
- 原始 URL 或数据集提供的来源标识。
- 下载时间、文件校验信息、过滤数量和抽样种子。
- 数据移除或重新生成机制。

若 tokenizer 或后续模型计划商业发布，应在冻结正式训练语料前单独完成许可证和适用法律审查。

## 主要依据

- [HPLT 3.0 官方数据页](https://hplt-project.org/datasets/v3.0)
- [HPLT 3.0 Hugging Face 数据卡](https://huggingface.co/datasets/HPLT/HPLT3.0)
- [FineWeb2 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- [FineWeb 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [Wikimedia Dumps 许可证说明](https://dumps.wikimedia.org/legal.html)
- [CulturaX 数据卡](https://huggingface.co/datasets/uonlp/CulturaX)
- [MADLAD-400 数据卡](https://huggingface.co/datasets/allenai/MADLAD-400)
- [TensorFlow Datasets mC4 说明](https://www.tensorflow.org/datasets/catalog/c4)

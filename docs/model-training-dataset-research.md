# MVP 模型训练数据集调研与来源锁定

状态：TD-02 source research / locked

调研日期：2026-07-15

## 结论

首轮有界人类平行数据只锁定官方 **MASSIVE 1.1**。它是由专业译者将 English SLURP seed 本地化到多种 locale 的多平行语料；官方论文将其描述为跨 51 种语言的平行、标注虚拟助手 utterance，并说明 50 个非英语版本由专业译者本地化。[ACL 论文与摘要](https://aclanthology.org/2023.acl-long.235/)

官方 1.1 发布同时包含 `en-US`、`zh-CN`、`zh-TW`、`ja-JP`、`ko-KR`，因此一个 `(partition, id)` group 可以在不做脚本推断、简繁转换或跨来源拼接的情况下形成项目所需的 9 个无向组。官方仓库明确提供 1.1 S3 归档、JSONL 结构、`train/dev/test` 字段和 `utt` 原文含义。[MASSIVE 官方仓库的数据说明](https://github.com/alexa/massive#accessing-and-processing-the-data)

数据归档内的 `LICENSE` 是 CC BY 4.0；`NOTICE.md` 说明 English 数据来自同为 CC BY 4.0 的 SLURP。归档、许可、notice 和五个 locale 文件的实际身份已写入 [`mvp_model_data.lock.json`](../configs/mvp_model_data.lock.json)。对外再分发数据或其改编版本时必须保留归属、许可链接和修改说明；模型许可不能替代数据许可审查。

## 为什么适合本轮 MVP

- **脚本身份明确**：官方 locale 分别是 `zh-CN` 与 `zh-TW`，映射为 `zho_Hans` 与 `zho_Hant`；`zh-TW` 不是 `yue_Hant`，也不是项目运行时对 `zh-CN` 做的简转繁。
- **人工来源明确**：非英语 locale 是对同一 English seed 的专业人工本地化，不是 teacher 生成数据。
- **对齐键稳定**：五个选中 locale 的 `(partition, id)` 集合实测完全一致。
- **许可统一**：选中归档的 data/SLURP notice 都指向 CC BY 4.0。
- **范围有界**：单一 40,251,390-byte 归档即可覆盖全部 9 组，避免在 TD-02 引入大规模抓取。

局限也很明确：数据是单轮虚拟助手领域，句子较短；localization 允许按 locale 调整实体或 slot 值；日/韩/中文之间是通过同一 English seed 对齐，而不是每个非英语 pair 直接互译。因此它适合证明训练链路和语言控制，不足以支持生产翻译质量结论。

## 实测身份与结构

官方归档：`https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz`

| 项目 | 实测值 |
| --- | --- |
| 版本 | 1.1 |
| HTTP Content-Length | 40,251,390 bytes |
| SHA-256 | `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577` |
| ETag | `51e0da2a3ff7a016f109e1d1b4306e93-3` |
| Last-Modified | 2022-11-07T16:55:04Z |
| 选中数据+许可文件 | 51,782,238 bytes |
| 每 locale 总行数 | 16,521 |
| 每 locale partition | train 11,514 / dev 2,033 / test 2,974 |

官方 Hugging Face 数据卡的 summary 曾写每语言 19,521 条，但同一数据卡的 split 表、官方 1.1 JSONL 和本次逐行解析都给出 16,521（11,514 + 2,033 + 2,974）。本项目以锁定归档的实际字节与逐行统计为准，并在 lock 中记录该结果。[Hugging Face split 表](https://huggingface.co/datasets/AmazonScience/massive#data-splits)

五个 locale 文件均验证 locale 字段唯一正确、`(partition, id)` 唯一 16,521 个，且与 `en-US` 集合零差异。后续 TD-03 只读取 `utt`，不把带 slot 标注的 `annot_utt` 当普通翻译文本。

## 9 组覆盖矩阵

| 无向 pair | MASSIVE locale | train 原始上限 | dev 原始上限 | test 原始上限 | 来源类型 |
| --- | --- | ---: | ---: | ---: | --- |
| `eng_Latn--jpn_Jpan` | `en-US` / `ja-JP` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `eng_Latn--kor_Hang` | `en-US` / `ko-KR` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `jpn_Jpan--kor_Hang` | `ja-JP` / `ko-KR` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `eng_Latn--zho_Hans` | `en-US` / `zh-CN` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `jpn_Jpan--zho_Hans` | `ja-JP` / `zh-CN` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `kor_Hang--zho_Hans` | `ko-KR` / `zh-CN` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `eng_Latn--zho_Hant` | `en-US` / `zh-TW` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `jpn_Jpan--zho_Hant` | `ja-JP` / `zh-TW` | 11,514 | 2,033 | 2,974 | human multiparallel localization |
| `kor_Hang--zho_Hant` | `ko-KR` / `zh-TW` | 11,514 | 2,033 | 2,974 | human multiparallel localization |

每组最低 accepted 门槛冻结为 train 10,000、dev 1,500、test 2,500；扫描上限为每 locale 16,521 行，下载上限为归档精确大小，选中解压上限为 52,000,000 bytes。若 TD-03/TD-05 清洗后任一组低于门槛，必须回到新的 source research 决策，不能静默用重复、简繁转换或 teacher output 补足 human dev/test。

## 繁体边界

`zh-TW -> zho_Hant` 的依据是官方 locale 和人工本地化流程，不是字符级自动分类。即便简繁共享大量字符，TD-03/TD-05 仍须做脚本合规与人工抽检；locale 证据不能替代内容质量检查。

- 禁止将 `zh-CN` 自动转换后标记为原生 `zho_Hant`。
- 禁止将 FLORES 的 `yue_Hant` 或任何粤语繁体数据映射为普通话 `zho_Hant`。
- 禁止把只有 `zh` 标签、没有来源地区/脚本证据的数据静默放入任一中文桶。
- teacher synthetic 只能进入后续显式 provenance 链，不能替代本数据集的人类 dev/test。

## 未选候选与原因

| 候选 | 结论 | 原因 |
| --- | --- | --- |
| Hugging Face `AmazonScience/massive` 浮动 `main` | 不作为 source identity | 页面和 parquet 转换可继续更新，且数据卡注明上传集成人员不是原 corpus 作者；只用它交叉核对结构，正式 lock 使用官方 S3 1.1 归档。 |
| FLORES-200 | 保留为未来独立评测研究，不进入本轮 train | 官方定位是 MT evaluation benchmark，只有 dev/devtest/hidden test；将其混入 train 会污染后续标准评测。[官方 FLORES-200 README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) |
| OPUS 聚合/`latest` 查询 | 本轮排除 | OPUS 是大量不同来源和许可证的集合，必须逐 corpus/版本审查；浮动聚合不能作为一个统一许可 source lock。[OPUS corpus 目录](https://opus.nlpl.eu/corpora) |
| HPLT 3.0 monolingual corpus | 不作为人类平行来源 | 当前冻结 HPLT 数据没有跨语言 alignment key，只能服务 tokenizer/未来单语增强。 |
| 自动简繁转换或未锁定 LLM 生成 | 不作为 human source | 不能证明原生繁体，也不满足本 task 的人工平行与完整 provenance 要求。 |

当前 9 组不存在必须用 synthetic 才能关闭的来源缺口，因此 TD-02 不启用 synthetic 补充。后续若清洗使 accepted 数低于门槛，应新立来源变更并更新 config hash/source lock；不得在现有 lock 下临时扩源。

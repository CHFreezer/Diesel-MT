# MVP 模型训练数据集调研与来源锁定

状态：TD-02 `completed`；10 组 schema v2 source/lock 已冻结

调研日期：2026-07-15

## 结论

首轮有界人类平行数据只锁定官方 **MASSIVE 1.1**。它是由专业译者将 English SLURP seed 本地化到多种 locale 的多平行语料；官方论文将其描述为跨 51 种语言的平行、标注虚拟助手 utterance，并说明 50 个非英语版本由专业译者本地化。[ACL 论文与摘要](https://aclanthology.org/2023.acl-long.235/)

官方 1.1 发布同时包含 `en-US`、`zh-CN`、`zh-TW`、`ja-JP`、`ko-KR`，因此一个 `(partition, id)` group 可以在不做脚本推断、自动简繁转换或跨来源拼接的情况下形成项目所需的 10 个无向模型关系：9 个跨语言关系和 `zh-CN--zh-TW` 中文内部本地化关系。官方仓库明确提供 1.1 S3 归档、JSONL 结构、`train/dev/test` 字段和 `utt` 原文含义。[MASSIVE 官方仓库的数据说明](https://github.com/alexa/massive#accessing-and-processing-the-data)

数据归档内的 `LICENSE` 是 CC BY 4.0；`NOTICE.md` 说明 English 数据来自同为 CC BY 4.0 的 SLURP。归档、许可、notice 和五个 locale 文件的实际字节身份已经锁定，第 10 组复用同一锁定字节且无需重新下载。[`mvp_model_data.lock.json`](../configs/mvp_model_data.lock.json) 已绑定 10 组 schema v2 配置哈希 `1c3fda336a5fae183ea48e813c442daabee5b754bfbd792bad15fabaeb2c52b7`。对外再分发数据或其改编版本时必须保留归属、许可链接和修改说明；模型许可不能替代数据许可审查。

## 为什么适合本轮 MVP

- **脚本身份明确**：官方 locale 分别是 `zh-CN` 与 `zh-TW`，映射为 `zho_Hans` 与 `zho_Hant`；`zh-TW` 不是 `yue_Hant`，也不是项目运行时对 `zh-CN` 做的简转繁。
- **人工来源明确**：非英语 locale 是对同一 English seed 的专业人工本地化，不是 teacher 生成数据。
- **对齐键稳定**：五个选中 locale 的 `(partition, id)` 集合实测完全一致。
- **许可统一**：选中归档的 data/SLURP notice 都指向 CC BY 4.0。
- **范围有界**：单一 40,251,390-byte 归档即可覆盖全部 10 组，避免在 TD-02 引入大规模抓取。

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

## 10 组覆盖矩阵

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
| `zho_Hans--zho_Hant` | `zh-CN` / `zh-TW` | 11,514 | 2,033 | 2,974 | human Chinese-internal localization |

每组最低 accepted 门槛冻结为 train 10,000、dev 1,500、test 2,500；扫描上限为每 locale 16,521 行，下载上限为归档精确大小，选中解压上限为 52,000,000 bytes。若 TD-03/TD-05 清洗后任一组低于门槛，必须回到新的 source research 决策，不能静默用重复、简繁转换或 teacher output 补足 human dev/test。

## 繁体边界

`zh-TW -> zho_Hant` 的依据是官方 locale 和人工本地化流程，不是字符级自动分类。模型与 teacher 语言名称仍为 `Traditional Chinese`；`zh-TW` 只记录当前人类来源及其用词偏向。即便简繁共享大量字符，TD-03/TD-05 仍须做脚本、语义保持与人工抽检；locale 证据不能替代内容质量检查。

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

当前 10 组都由同一锁定 MASSIVE 归档提供，不需要 synthetic 才能关闭来源缺口。第 10 组已经进入新 config hash/source lock 与覆盖统计；归档 URI、成员大小和 SHA-256 均未改变。后续若清洗使 accepted 数低于门槛，应新立来源变更；不得临时改义。

## TD-05 独立评测污染引用结论

TD-05 最终选择 **原版 FLORES-200** 的 `dev`/`devtest` 作为外部污染阻断引用，不选择持续更新的 FLORES+。这里的“使用”只表示在构建 M0 时检查训练候选是否与评测文本精确或近重复，绝不表示把 FLORES 文本并入 MASSIVE、训练 split、方向采样或 teacher 输入。

选择原版的理由是身份更适合当前可复现门禁：Meta 官方仓库已经归档，固定 revision `a6c830c6e1051fb4ac1a44b32358f00463f332bd` 明确列出项目五个标签并指向固定的 2022 数据归档；FLORES+ 则是仍在维护、按版本扩展的后继集合，适合后续另立评测升级决策，但不适合作为本次 M0 中途漂移的引用。[原版官方 README](https://github.com/facebookresearch/flores/blob/a6c830c6e1051fb4ac1a44b32358f00463f332bd/flores200/README.md)；[FLORES+ 数据卡](https://huggingface.co/datasets/openlanguagedata/flores_plus)

冻结身份位于 [`mvp_mt_evaluation.lock.json`](../configs/mvp_mt_evaluation.lock.json)：

- 官方归档 `flores200_dataset.tar.gz`：25,585,843 bytes，SHA-256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`；
- `eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang` 的 `dev` 997 行与 `devtest` 1,012 行，共 10,045 个单语引用记录；
- 仓库 README、benchmark README 和 `LICENSE_CC-BY-SA` 均锁定大小/SHA-256；
- `reference-manifest.json` 明确标记 `prohibited from training`，污染 registry 对它使用 `policy=block` 与 `match=exact-and-near`。

M0 正式扫描结果为 FLORES-200 `hits=0`。未来 TD-13 若真正用该集合计算模型质量，必须继续读取同一锁定身份；若改用 FLORES+，应作为显式评测版本升级，不能静默替换本锁。

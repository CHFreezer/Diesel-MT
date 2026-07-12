# plan: tokenizer dataset fetch script

## 目标

编写 tokenizer 训练所需数据集的获取脚本，为 MVP tokenizer 提供可复现、可审计、可扩展的中英日韩文本语料输入。

该脚本的核心目标不是一次性下载尽可能多的数据，而是建立稳定的数据入口：来源明确、许可证可检查、版本可锁定、处理结果可复现、语料规模可按 MVP 资源限制调整。

## 范围

本 plan 覆盖 tokenizer 语料获取脚本的边界、数据目录约定、来源登记、下载行为、清洗抽样、manifest 和验证标准。

本 plan 不覆盖 tokenizer 训练逻辑，不覆盖翻译平行语料训练集构建，也不覆盖 Hy-MT2 蒸馏样本生成。

## 约束

- 脚本只为 tokenizer 训练准备文本语料，不直接产出模型训练样本。
- 数据源必须有明确来源、许可证、下载地址或数据集标识、版本信息和语言标注。
- 默认不接受许可证不明、禁止再利用或仅非商业使用的数据源。
- 原始下载数据、解压中间文件和最终语料默认不提交到 Git；仓库只提交脚本、配置、manifest 示例和小型测试样本。
- 脚本必须支持断点续跑、跳过已完成下载、校验文件大小或 checksum，并能在无网络时使用本地缓存重新生成处理后语料。
- 清洗策略必须保守，不能改变语言本身形态；只做编码修复、空白规范化、明显坏行过滤、去重和抽样。
- 稳定复现以输出文件字节级一致为标准，不能只比较行数、字符数等聚合统计。

## 调研依据

数据源选型、排序、限制和 MVP 推荐组合见 [中英日韩 tokenizer 数据集调研](../../docs/tokenizer-dataset-research.md)。实现 registry 时应以该文档的推荐方案为默认值，并在锁定数据版本前复核官方数据卡和许可证条款。

## 执行拆解

具体待办事项见 [tokenizer dataset fetch script todo](../todo/tokenizer-dataset-fetch-script.md)。

## 稳定复现契约

在 source lock、项目配置、代码版本、依赖锁和运行 profile 相同的前提下，不同运行生成的四个语料文件和确定性 manifest 必须字节级一致，SHA-256 必须相同。

实现必须满足：

- 下载前先解析数据源并生成 source lock。lock 固定 map 文件哈希、输入分片 URL、逻辑顺序、文件大小和已知或首次下载后计算的 SHA-256。
- 正式构建只能消费已有 lock，不能隐式读取 `latest`、重新解析远端 map 或静默接受变化后的分片。
- 缓存文件每次使用前都按 lock 校验；文件损坏、远端内容变化或校验缺失时必须明确失败。
- 输入按 lock 中的逻辑顺序处理。并发下载或处理的完成顺序不能改变合并顺序和最终输出。
- 抽样使用固定种子和稳定内容标识；不得使用受进程随机化影响的 Python 内置 `hash()`。
- 清洗、精确去重、MinHash 参数、哈希算法、长度边界和字符预算全部由版本化配置明确给出。
- 输出统一为 UTF-8 无 BOM 和 LF 换行。除显式、版本化的空白规则外，不做依赖平台或 locale 的隐式文本转换。
- `manifest.jsonl` 只记录决定内容的稳定字段，并使用固定字段顺序和排序规则。下载时间、耗时、机器路径等易变字段写入独立的运行记录，不能影响内容哈希。
- manifest 记录配置哈希、source lock 哈希、Git commit、工作树 dirty 状态、相关源码哈希、Python 版本、依赖锁哈希、算法版本和每个输出文件的 SHA-256。

## 数据源登记

脚本应使用显式数据源 registry，而不是把 URL 和处理规则散落在代码里。registry 至少记录：

```text
source_id
source_type
license
homepage
download_uri
version_or_snapshot
languages
expected_files
checksum_or_size
enabled
notes
```

MVP 数据源应优先选择许可证清晰、下载方式稳定、语言覆盖明确的公开文本或平行语料来源。对于平行语料，可抽取两侧文本作为 tokenizer 语料；对于单语语料，必须通过语言标注或语言识别确认文本语言。

## 目录约定

建议脚本输出目录如下：

```text
data/tokenizer/
  raw/
  cache/
  interim/
  corpus/mvp/
    eng_Latn.txt
    zho_Hans.txt
    jpn_Jpan.txt
    kor_Hang.txt
    manifest.jsonl
  reports/
```

`raw` 保存原始下载文件，`cache` 保存可复用缓存，`interim` 保存解压或解析后的中间结果，`corpus/mvp` 保存 tokenizer 训练入口文本，`reports` 保存统计报告。

## 脚本形态

MVP 脚本建议放在：

```text
scripts/fetch_tokenizer_datasets.py
```

配置建议放在：

```text
configs/tokenizer_datasets_mvp.yaml
```

CLI 需要支持以下语义：

```text
python scripts/fetch_tokenizer_datasets.py --config configs/tokenizer_datasets_mvp.yaml --out data/tokenizer
python scripts/fetch_tokenizer_datasets.py --config configs/tokenizer_datasets_mvp.yaml --dry-run
python scripts/fetch_tokenizer_datasets.py --config configs/tokenizer_datasets_mvp.yaml --use-cache
```

## 处理边界

脚本可做的处理：

- 下载、校验、解压和缓存数据文件。
- 从公开语料中抽取中英日韩文本列。
- 过滤空行、过短行、过长行、明显乱码行、HTML 残留和重复行。
- 按语言进行均衡抽样，避免某一种语言压倒词表训练。
- 生成每行来源记录或聚合 manifest，保证后续可追溯。

脚本不应做的处理：

- 不做机器翻译、改写或蒸馏。
- 不做英文大小写折叠。
- 不做中文简繁转换。
- 不做日文、韩文转写。
- 不混入第三方 tokenizer 文件或模型资产。

## 产物

脚本执行后应产出：

- `eng_Latn.txt`、`zho_Hans.txt`、`jpn_Jpan.txt`、`kor_Hang.txt` 四个 tokenizer 训练文本文件。
- `manifest.jsonl`，记录每个输出文件对应的数据源、版本、许可证、样本数、字符数、过滤数量和抽样种子。
- `reports/tokenizer_corpus_mvp.md`，记录每种语言的行数、字符数、去重比例、长度分布和来源占比。
- 可选的小型 fixture 数据，用于在 CI 或本地快速验证脚本逻辑。

## 验证

必须验证：

- `--dry-run` 能列出将要下载和处理的数据源，不产生大文件。
- 使用同一 source lock、配置、代码和依赖锁，在两个全新输出目录重复运行，语料文件与确定性 manifest 的 SHA-256 一致。
- 输出四种语言文件均非空，且语言标注和文件名一致。
- manifest 能追溯到每个启用数据源的许可证和版本。
- 缓存存在时可跳过网络下载并重新生成最终语料。
- 并发度变化和缓存命中状态变化不影响最终输出顺序与文件哈希。
- 远端 map、分片内容或缓存校验与 source lock 不一致时明确失败，不静默生成新结果。
- 失败时给出明确错误信息，不留下被误认为成功的半成品语料。

## 验收标准

- 脚本能生成 MVP tokenizer 所需四语种语料入口。
- 所有启用数据源都有来源、许可证和版本记录。
- 输出语料具备基础质量报告和可复现 manifest。
- source lock、配置指纹、依赖指纹和最终输出 SHA-256 齐全。
- 两次独立构建通过字节级复现验证。
- 脚本默认行为不会把大体积数据写入 Git 跟踪路径。
- 生成的 `data/tokenizer/corpus/mvp/` 可直接作为 MVP tokenizer 训练输入。

## 风险

- 数据源下载地址或数据集版本可能变化，需要 registry 和 manifest 把版本锁住。
- 不同语料许可证可能不兼容，需要在启用前做明确筛选。
- 日文、韩文公开语料规模可能小于英文、中文，需要抽样策略保证 tokenizer 不偏科。
- 语言识别工具可能误判短句或混合文本，MVP 应保留人工抽样检查入口。

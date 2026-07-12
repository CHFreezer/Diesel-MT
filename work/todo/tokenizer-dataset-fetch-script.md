# todo: tokenizer dataset fetch script

## 来源

- plan：[tokenizer dataset fetch script](../plan/tokenizer-dataset-fetch-script.md)
- 调研：[中英日韩 tokenizer 数据集调研](../../docs/tokenizer-dataset-research.md)
- 环境：[Python 环境约定](../../docs/python-environment.md)
- task 索引：[tokenizer dataset fetch script tasks](../task/tokenizer-dataset-fetch-script/index.md)

## 目标

实现并运行 tokenizer 数据集获取脚本，先完成小规模 smoke 数据闭环，再生成可直接训练 MVP tokenizer 的中英日韩均衡语料。

本 todo 只拆解数据获取、清洗、抽样和验收事项，不包含 tokenizer 训练和翻译模型训练。

## 已确定决策

- 主数据源使用 HPLT 3.0。
- 语言映射为 `eng_Latn -> eng_Latn`、`cmn_Hans -> zho_Hans`、`jpn_Jpan -> jpn_Jpan`、`kor_Hang -> kor_Hang`。
- 优先抽取 HPLT 3.0 的 WDS 10 至 8 高质量分片。
- smoke 阶段只使用 HPLT 3.0，不混入 Wikimedia 或 FineWeb。
- 简体中文不做简繁转换，不默认引入中文 Wikimedia。
- 语料按清洗后 Unicode 字符量均衡，不按文档数均衡。
- 英文和中文必须在项目侧补充去重。
- 所有 Python 命令在项目 `.conda` 环境中执行。
- 稳定复现以最终语料和确定性 manifest 的 SHA-256 一致为验收标准。

## 待办

### [TD-01 冻结下载配置](../task/tokenizer-dataset-fetch-script/td-01-freeze-download-config.md)

- [ ] 确认 HPLT 3.0 四种语言 map URL 当前可访问。
- [ ] 在配置中固定数据集版本、官方主页、许可证说明和下载入口。
- [ ] 定义 `smoke` 与 `mvp` 两个 profile 的字符预算、随机种子和启用语言。
- [ ] 定义复现边界：source lock、配置、代码版本、依赖锁和 profile 相同时输出必须字节级一致。
- [ ] `smoke` profile 使用足够小的预算，以便一次本地运行快速完成。
- [ ] `mvp` profile 目标为每种语言 10 亿至 20 亿个清洗后 Unicode 字符；执行前核对磁盘和网络预算。

产物：`configs/tokenizer_datasets_mvp.yaml`。

### [TD-02 建立目录与 Git 边界](../task/tokenizer-dataset-fetch-script/td-02-data-layout-and-git-boundary.md)

- [ ] 创建 `data/tokenizer/raw/`、`cache/`、`interim/`、`corpus/mvp/` 和 `reports/` 的运行时目录。
- [ ] 更新 `.gitignore`，排除原始数据、缓存、中间文件和生成语料。
- [ ] 保留可提交的小型 fixture 目录，用于无网络测试。
- [ ] 建立并提交依赖声明及依赖 lock，固定直接和传递依赖版本，并记录 lock 文件 SHA-256。
- [ ] 确认 `git status` 不会枚举大体积数据文件。

产物：目录约定、`.gitignore` 规则和 fixture 目录。

### [TD-03 建立数据源 registry](../task/tokenizer-dataset-fetch-script/td-03-source-registry-and-lock.md)

- [ ] 定义并校验 `source_id`、`source_type`、`license`、`homepage`、`download_uri`、`version_or_snapshot`、`languages`、`expected_files`、`checksum_or_size`、`enabled` 和 `notes` 字段。
- [ ] 为 HPLT 3.0 四种语言登记独立来源项和语言映射。
- [ ] 将 Wikimedia、FineWeb 和 FineWeb2 登记为默认禁用的备用来源。
- [ ] 对缺少许可证、版本或下载入口的数据源执行快速失败。
- [ ] 定义 source lock 格式，固定 map 哈希、分片 URL、逻辑顺序、大小和 SHA-256。
- [ ] 将不包含凭据和机器路径的 source lock 作为可审计配置保存。

产物：可由脚本读取和验证的数据源配置，以及 `configs/tokenizer_datasets_mvp.lock.json`。

### [TD-04 实现 CLI 与 dry-run](../task/tokenizer-dataset-fetch-script/td-04-cli-and-dry-run.md)

- [ ] 创建 `scripts/fetch_tokenizer_datasets.py`。
- [ ] 支持 `--config`、`--lock`、`--out`、`--profile`、`--dry-run`、`--use-cache`、`--offline` 和随机种子参数。
- [ ] 提供显式的 source lock 解析命令；正常构建不得隐式更新 lock。
- [ ] `--dry-run` 输出数据源、语言映射、WDS 范围、字符预算、缓存位置和预计操作。
- [ ] `--dry-run` 不下载或创建大文件。
- [ ] 参数和配置错误返回非零退出码及可定位的错误信息。

产物：可执行 CLI 和 dry-run 输出。

### [TD-05 实现 HPLT 3.0 获取器](../task/tokenizer-dataset-fetch-script/td-05-hplt-fetcher.md)

- [ ] 读取每种语言的 HPLT map 文件并解析 `.jsonl.zst` 分片 URL。
- [ ] 只选择配置允许的高质量 WDS 分片。
- [ ] 支持流式读取，达到字符预算后停止，不能默认下载完整语言语料。
- [ ] 构建阶段严格按 source lock 的逻辑顺序读取和合并，结果不受并发完成顺序影响。
- [ ] 支持下载重试、超时、断点续跑和已完成分片跳过。
- [ ] 缓存 map 文件、数据分片状态和 SHA-256；每次复用缓存前重新校验。
- [ ] source lock 与远端或缓存内容不一致时快速失败，不能静默接受新内容。
- [ ] `--use-cache` 模式下不访问网络也能重新处理已有数据。

产物：HPLT 3.0 下载与缓存实现。

### [TD-06 实现文本抽取与保守清洗](../task/tokenizer-dataset-fetch-script/td-06-text-extraction-and-cleaning.md)

- [ ] 从 HPLT 文档中抽取正文，保留来源 URL 和语言元数据。
- [ ] 统一 UTF-8 编码和空白，但不改变语言文字形态。
- [ ] 过滤空行、明显乱码、HTML 残留和超出配置长度范围的文本。
- [ ] 不做英文小写化、中文简繁转换、日文假名转换或韩文罗马化。
- [ ] 记录每类过滤规则的输入数、保留数和拒绝数。

产物：四语清洗后中间文本与过滤统计。

### [TD-07 实现去重与均衡抽样](../task/tokenizer-dataset-fetch-script/td-07-dedup-and-balanced-sampling.md)

- [ ] 对四种语言执行行级或段落级精确去重。
- [ ] 对英文和中文执行 MinHash 近似去重，并固定参数。
- [ ] 按语言字符预算停止抽样，保证四种语言的最终规模可比较。
- [ ] 固定抽样种子，避免处理顺序改变结果。
- [ ] 使用稳定内容标识和明确哈希算法，不使用 Python 内置 `hash()`。
- [ ] 固定 MinHash 的分词单位、n-gram、哈希函数、阈值、seed 和并列项排序规则。
- [ ] 保存去重前后数量、字符量和重复率。

产物：可复现的四语均衡样本。

### [TD-08 生成训练入口与 manifest](../task/tokenizer-dataset-fetch-script/td-08-corpus-and-manifest.md)

- [ ] 生成 `eng_Latn.txt`、`zho_Hans.txt`、`jpn_Jpan.txt` 和 `kor_Hang.txt`。
- [ ] 每行保持单个训练文本单元，不写入来源标记或语言 token。
- [ ] 输出统一使用 UTF-8 无 BOM 和 LF 换行，固定排序和尾部换行规则。
- [ ] 生成确定性 `manifest.jsonl`，记录来源、版本、许可证、语言映射、样本数、字符数、过滤数、去重数和抽样种子。
- [ ] manifest 记录配置、source lock、Git commit、dirty 状态、相关源码和依赖锁哈希，以及每个输出文件的 SHA-256。
- [ ] 将下载时间、耗时、绝对路径等易变信息写入独立运行记录，不写入确定性 manifest。
- [ ] 采用临时文件加原子重命名，失败时不遗留看似完整的最终文件。
- [ ] 对最终文件记录 SHA-256 或等价稳定校验值。

产物：`data/tokenizer/corpus/mvp/` 下的训练入口和 manifest。

### [TD-09 生成质量报告](../task/tokenizer-dataset-fetch-script/td-09-quality-report.md)

- [ ] 统计每种语言的行数、字符数、长度分布和来源占比。
- [ ] 统计精确去重率、近似去重率和各过滤规则命中率。
- [ ] 对每种语言固定抽取人工检查样本，不包含到报告正文中的敏感原文。
- [ ] 标记语言混入、乱码、模板文本或异常重复等检查结果。

产物：`data/tokenizer/reports/tokenizer_corpus_mvp.md`。

### [TD-10 建立自动化测试](../task/tokenizer-dataset-fetch-script/td-10-automated-tests.md)

- [ ] 为 registry 校验、map 解析、语言映射、清洗、去重和字符预算编写单元测试。
- [ ] 使用小型 fixture 验证四种语言，不依赖网络或完整数据集。
- [ ] 测试同一配置和种子重复运行得到一致统计与校验值。
- [ ] 在两个全新输出目录执行同一构建，逐文件比较字节和 SHA-256。
- [ ] 使用不同并发度、一次冷缓存和一次热缓存构建，验证输出哈希不变。
- [ ] 测试 source lock、远端分片或缓存校验不一致时构建明确失败。
- [ ] 测试缓存复用、网络失败和中途失败恢复。
- [ ] 测试输出文件非空且语言与文件名一致。

产物：可在本地快速执行的测试集。

### [TD-11 执行 smoke 下载](../task/tokenizer-dataset-fetch-script/td-11-smoke-download.md)

- [ ] 激活项目 `.conda` 环境并安装已声明依赖。
- [ ] 运行 `smoke` profile 的 `--dry-run` 并保存结果摘要。
- [ ] 执行四语 smoke 下载和处理。
- [ ] 在断网或禁用网络条件下使用缓存重复生成输出。
- [ ] 人工抽查每种语言样本和质量报告。
- [ ] 记录运行时间、下载量、峰值磁盘占用和发现的问题。

验收：四个 smoke 语料文件、确定性 manifest 和报告完整生成，两次独立运行逐文件 SHA-256 一致。

### [TD-12 执行 MVP 下载](../task/tokenizer-dataset-fetch-script/td-12-mvp-download.md)

- [ ] 根据 smoke 结果修正配置和过滤阈值，但不改变文字形态。
- [ ] 执行前确认 `mvp` profile 的预计下载量、可用磁盘空间和运行时间。
- [ ] 执行完整 MVP 下载、清洗、去重和均衡抽样。
- [ ] 复核 manifest 中所有启用来源的版本和许可证记录。
- [ ] 复核四语字符预算、语言质量、去重率和最终校验值。
- [ ] 在独立输出目录执行第二次构建并完成字节级复现对比。
- [ ] 将最终运行命令、配置版本和结果摘要写入报告。

验收：`data/tokenizer/corpus/mvp/` 可直接作为 MVP tokenizer 训练输入。

## 完成条件

- [ ] `--dry-run`、`--use-cache` 和正常下载路径均通过验证。
- [ ] 四种语言的最终语料、manifest 和质量报告齐全。
- [ ] 同一 source lock、配置、代码和依赖锁可以复现字节级一致的输出。
- [ ] 冷缓存、热缓存和不同并发度下输出 SHA-256 一致。
- [ ] 数据来源、版本、许可证和语言映射均可追溯。
- [ ] 大体积数据未进入 Git 跟踪范围。
- [ ] 结果经过人工抽样复核，可以进入 tokenizer 训练 task。

# task TD-01: 冻结下载配置

状态：done

依赖：无

## 目标

冻结 HPLT 3.0 数据源选择、语言映射、质量范围和运行 profile，为后续实现提供唯一配置入口。

## 输入

- [数据获取 todo](../../todo/tokenizer-dataset-fetch-script.md)
- [数据集调研](../../../../docs/tokenizer-dataset-research.md)
- HPLT 3.0 四种语言的官方 map 地址。

## 执行事项

- 验证 `eng_Latn`、`cmn_Hans`、`jpn_Jpan`、`kor_Hang` map 地址可访问。
- 固定 HPLT 版本、主页、许可证、下载入口和 WDS 10 至 8 范围。
- 定义 `smoke` 和 `mvp` profile 的语言、字符预算、随机种子和并发默认值。
- 明确 `cmn_Hans -> zho_Hans` 映射，其余语言保持同名映射。
- 将复现边界写入配置说明：source lock、配置、代码版本、依赖 lock 和 profile 相同时输出字节级一致。
- 记录 MVP 预计下载量与磁盘预算的计算方式，不在配置中依赖本机绝对路径。

## 产物

- `configs/tokenizer_datasets_mvp.yaml`
- 配置字段说明和默认值。

## 验收

- 配置可被 YAML 解析器加载，字段无隐式默认或歧义。
- 四种语言和两个 profile 均有明确字符预算。
- 配置不包含凭据、临时 URL、绝对路径或动态日期。
- 官方 map 地址验证结果和验证日期有记录。

## 验证记录（2026-07-13）

- 四个官方 map 地址均已访问并解析：`eng_Latn.map`、`cmn_Hans.map`、`jpn_Jpan.map`、`kor_Hang.map`，基础地址为 `https://data.hplt-project.org/three/sorted/`。
- 已冻结 HPLT 3.0（July 2025 release）、WDS 10 至 8、四个语言映射、seed `20260713`；实际 lock 只选取各语言首个 WDS 10 分片的锁定前缀。
- `smoke` 为每语言 200,000 字符，`mvp` 为每语言 1,000,000,000 字符；配置不含本机路径或凭据。
- 产物：`configs/tokenizer_datasets_mvp.yaml`，SHA-256 `9822f53542a5947c541a2a30094988d1acd9c0a9f01c83ef765a2197087488e4`。

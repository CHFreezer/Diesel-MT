# task TD-08: 产物打包与文档

状态：done

依赖：TD-07（MVP 默认选定）

## 目标

为 MVP 默认候选生成最终产物目录，包含规范 `tokenizer.json`、完整配置、映射文件和文档；非默认候选保留为备选。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- TD-06 保存的 tokenizer 目录
- TD-05 覆盖率报告
- TD-07 选定记录和对比报告
- TD-04 语言 token 映射 JSON
- TD-03 训练配置和语料 manifest 快照

## 执行事项

- 为 MVP 默认候选生成最终产物目录，包含：
  - 规范 `tokenizer.json`
  - `tokenizer_config.json` 和必要的 special token 配置
  - 语言 token → ID 映射（JSON）
  - 训练配置快照（参数、种子、语料 manifest 引用、锁定依赖版本）
  - 覆盖率报告
  - 候选对比报告和选定记录
  - 最小编码样例（四语 test case，JSONL 格式）
- 为非默认候选保留产物，标注为备选，包含其独立的覆盖率报告和配置快照。
- 编写 `artifacts/tokenizers/README.md`，说明：
  - 目录结构和各文件用途
  - 复现步骤（锁定版本、命令行、语料 manifest 引用）
  - MVP 默认候选和备选的差异
  - 已知限制和后续改进方向
- 确认产物目录不含绝对路径、本机机器名、用户名或临时目录引用。
- 确认产物不包含 NLLB-200、M2M100 或任何第三方 tokenizer 资产。

## 产物

- `artifacts/tokenizers/mvp-32k/` 完整目录（如是默认候选；否则为备选）
- `artifacts/tokenizers/mvp-48k/` 完整目录（如是默认候选；否则为备选）
- `artifacts/tokenizers/README.md`

## 验收

- 产物目录包含所有列出文件，无不完整项。
- `README.md` 包含可执行的复现步骤。
- 无绝对路径、机器名或第三方 tokenizer 资产。
- 非默认候选已标注并保留。

## 验证记录

2026-07-14 已发布 `artifacts/tokenizers/mvp-tokenizer-v0/`，包含规范 tokenizer/config、五语映射、训练 metadata、alphabet audit、训练 corpus manifest 快照和逐文件 SHA manifest。`artifacts/tokenizers/README.md` 记录目录语义、复现命令、冻结哈希、32k 回退和已知限制；固定评测与冻结报告位于 `artifacts/tokenizers/reports/mvp-tokenizer-v0/`。冻结根 SHA-256：`eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。

# task TD-11: 自动化测试

状态：done（tokenizer 冻结范围）

依赖：TD-01（环境）可建框架；全部模块测试在 TD-03 至 TD-10 完成后收口

## 目标

建立 `tests/test_tokenizer.py` 测试集，覆盖训练复现性、tokenizer 行为正确性、词表/ID 一致性、边界用例、语言裁剪和 CTranslate2 集成冒烟。测试可在无网络条件下运行。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- TD-01 锁定的依赖版本
- TD-03 训练脚本和候选 tokenizer
- TD-04 语言 token 映射和 allowlist
- TD-05 评测样本集
- TD-06 保存/重载产物
- TD-09 集成验证脚本
- TD-10 CTranslate2 冒烟产物

## 执行事项

本任务从 TD-01 完成后即可开始搭建测试框架，随后按模块完成顺序逐步补充测试用例。最终收口在 TD-10 完成之后。

### 早期（TD-01 完成后）
- 搭建 `tests/test_tokenizer.py` 和 test fixtures 目录。
- 准备固定的测试 fixture（小规模四语样例、边缘文本、特殊字符集）。
- 实现基础 fixture 加载工具函数。

### 中期（TD-03 完成后逐步补充）
- 测试固定输入、采样种子、批次顺序和依赖版本下两次训练产物一致；若 JSON 仅序列化顺序不同，则比较规范化 JSON 和 encode 行为。
- 测试四种语言样例的 encode/decode 往返正确性。
- 测试所有语言 token 不会被切分为多个子词（每个语言 token encode 后为单一 ID）。
- 测试五个保留语言全部存在且 ID 不等于 `<unk>`。
- 测试代表性的已裁剪语言（`fra_Latn`、`deu_Latn`、`rus_Cyrl`）不在词表中，并由 Python allowlist 在 token ID 转换前拒绝。
- 测试 ID 稠密且 `len(tokenizer)` 与 M2M100Config 的 `vocab_size`、embedding/projection 行数匹配。
- 测试 special token ID 映射稳定（`<s>=0, <pad>=1, </s>=2, <unk>=3`）。
- 测试保存前的 `NllbTokenizer` 与离线重载后的 `AutoTokenizer` 对相同文本的 token、ID 和 decode 结果一致。
- 测试训练前、训练后、离线重载和 CTranslate2 转换器加载边界的 tokenizer 均为 fast backend。
- 项目源码中不允许导入 `NllbTokenizerFast`（添加静态检查或 lint 规则）。

### 晚期（TD-05 至 TD-10 完成后补充）
- 添加边界测试：空字符串、纯空白、纯特殊 token、超长行（>10k 字符）、未知字符/emoji。
- 增加 alphabet 回归集：罕见汉字与姓名用字、平假名/片假名、Hangul 音节与 Jamo、ASCII/全角数字和标点、常见 emoji；对 must-cover 字符断言不产生 `<unk>`。
- 增加连续未知字符测试，验证质量报告按 offset 覆盖的原文字符数计数，而不是把 `fuse_unk=true` 生成的一个 `<unk>` 误计为仅丢失一个字符。
- 测试 `forced_bos_token_id` 对四个目标语言均可正确获取非零 ID。
- 添加 CTranslate2 转换与 CPU `target_prefix` 推理冒烟测试；可标记为独立的慢速集成测试（`pytest.mark.slow`），但属于发布前必跑项。

## 产物

- `tests/test_tokenizer.py`
- `tests/fixtures/tokenizer/` 目录（test fixtures）
- 测试运行报告

## 验收

- 所有测试可在无网络条件下通过 `python -m pytest tests/test_tokenizer.py -q` 运行。
- 训练复现性测试在同参数下两次训练产物 encode 行为一致。
- 语言 token 切分、ID 稳定性和 allowlist 拒绝测试全部通过。
- 边界测试覆盖空字符串、超长行、未知字符和连续未知字符。
- must-cover alphabet 回归测试中无 `<unk>`。
- CTranslate2 冒烟测试通过（可标记 slow，但必须可运行）。
- 全量测试无警告、无 skip（除显式标记 slow 的 CT2 测试外）。

## 验证记录

2026-07-14 新增五语数据/评测 fixture 和 `tests/test_tokenizer_freeze.py`，覆盖 artifact manifest 哈希、五语覆盖摘要与五方向微型 M2M100 forward。数据管线、训练、checkpoint、评测和冻结相关测试合计 `45 passed`。CTranslate2 慢速部署测试随 TD-10 一并 deferred，不属于本次有界 tokenizer 重训的完成条件。

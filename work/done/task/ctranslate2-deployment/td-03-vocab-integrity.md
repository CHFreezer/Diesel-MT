# task TD-03: 词表与 ID 空间完整性

状态：done

依赖：TD-02

## 目标

证明冻结 tokenizer、HF M2M100 checkpoint 和 CTranslate2 转换产物使用完全一致的 49,152 项 token ID 空间。

## 输入

- [CTranslate2 deployment todo](../../todo/ctranslate2-deployment.md)
- 冻结 tokenizer、TD-01 HF checkpoint、TD-02 float32/int8 产物

## 执行事项

- 从 tokenizer `get_vocab()` 构造严格按 ID 排序的 token 序列，并断言 ID 稠密唯一。
- 读取/导出 CT2 转换词表，逐 ID 与 tokenizer 比较，不只比较集合或长度。
- 核对 HF shared/encoder/decoder embedding 与 `lm_head` 行数。
- 核对五个语言 token 和 `<s>`、`<pad>`、`</s>`、`<unk>` 的 token、ID 及 CT2 配置语义。
- 对重复 token、ID 空洞、顺序差异、缺失 token 和配置错位提供明确错误信息。

## 验收

- tokenizer/HF/float32 CT2/int8 CT2 的完整 ID 空间一致。
- 所有核心和语言特殊 token 均存在、ID 正确且无 `<unk>` 退化。
- 机器可读校验报告记录词表大小、哈希和逐项比较结论。

## 实现记录（2026-07-14）

- 从冻结 tokenizer 构造 49,152 项按 ID 排序的唯一稠密 token 序列，序列 SHA-256 为 `72bc2edcfe44bdfac90d2c101f71f214cdd2c4b70d7c975e5d269011be40c716`。
- HF checkpoint tokenizer、float32 `shared_vocabulary.json` 和 int8 `shared_vocabulary.json` 均逐 ID 完全相同；不是只比较长度或集合。
- HF config/shared/encoder/decoder/`lm_head` 均为 49,152 行，输入输出 embedding 保持 tied。
- `<s>=0`、`<pad>=1`、`</s>=2`、`<unk>=3` 和五个 language token `5..9` 均通过；CT2 的 `decoder_start_token=</s>`，且不隐式添加 source BOS/EOS。
- 校验器对 ID 空洞、重复 ID/token、长度差异、首个错位、特殊 token 和 CT2 config 错位提供明确异常。机器可读记录见合并 [deployment-validation.json](../../../../artifacts/ctranslate2/deployment-validation.json) 的 `phases.td_03_vocab_integrity`。

本 task 已随整个 todo 通过统一 review 并归档。

# task TD-03: 词表与 ID 空间完整性

状态：pending

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

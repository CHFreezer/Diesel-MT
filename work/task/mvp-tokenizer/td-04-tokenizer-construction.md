# task TD-04: NllbTokenizer 构造与语言 token 映射

状态：pending

依赖：TD-01（训练环境与依赖）

## 目标

按锁定版本 Transformers 5.x 源码确定种子 `NllbTokenizer` 的构造方式、语言 token 注入接口和映射获取方法，验证五语言裁剪和 ID 空间完整性。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- TD-01 锁定的 `transformers` 版本及其 `tokenization_nllb.py` 源码
- TD-01 版本兼容记录

## 执行事项

- 按锁定的 Transformers 5.x 源码构造空 `NllbTokenizer`，仅传入五种语言 token；生产代码不得引用 4.x `NllbTokenizerFast` 或旧版 `vocab_file` 路线。
- 用锁定版本支持的 `extra_special_tokens` 或等价参数替换默认 200+ `FAIRSEQ_LANGUAGE_CODES`，仅保留 `eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang`。
- 保存后断言 `get_vocab()` 中五个保留语言均存在，抽取若干未保留 NLLB 语言（如 `fra_Latn`、`deu_Latn`、`rus_Cyrl`）断言均不存在。
- 实现 Python 应用层语言 allowlist 校验：在 `convert_tokens_to_ids()` 之前对未知语言码抛出明确错误，不能让未知语言码静默退化为 `<unk>`。
- 从保存后重载的 tokenizer 用 `convert_tokens_to_ids()` 生成语言 token → ID 映射，不依赖当前源码可能未提供的 `lang_code_to_id` 属性。
- 验证 `forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)` 对每个目标语言均可正确获取。
- 确认 `eos_token_id=2`、`pad_token_id=1`、`unk_token_id=3`，与 M2M100/NLLB 约定一致。
- 生成并保存语言 token → ID 的 JSON 映射文件。
- 验证最终总词表 ID 稠密、唯一（按 ID 排序后严格等于 `0..len(tokenizer)-1`）。
- 本任务定义种子 tokenizer 的构造规范；实际调用 `train_new_from_iterator()` 的训练流程在 TD-03。

## 产物

- 种子 `NllbTokenizer` 构造代码（供 TD-03 调用）
- 语言 token → ID 映射 JSON
- 语言 allowlist 校验函数
- special token ID 断言记录

## 验收

- 构造的 tokenizer 满足 `is_fast is True`。
- `get_vocab()` 中五个保留语言均存在且 ID ≠ `<unk>`。
- `fra_Latn`、`deu_Latn`、`rus_Cyrl` 不在 `get_vocab()` 中。
- 未知语言码在 `convert_tokens_to_ids()` 之前被 allowlist 拒绝并抛出明确错误。
- `forced_bos_token_id` 对四个目标语言可正确获取。
- `eos_token_id=2`、`pad_token_id=1`、`unk_token_id=3`。
- 最终词表 ID 稠密唯一，无空洞。
- 代码不引用 `NllbTokenizerFast` 或 4.x 旧接口。

## 验证记录

（待填写）

# task TD-09: 最小训练链路集成验证

状态：pending

依赖：TD-06（保存与加载验证）、TD-08（产物打包）

## 目标

用 MVP 默认 tokenizer 和随机初始化的微型 `M2M100ForConditionalGeneration` 跑通完整的 encoder 输入构造、decoder 目标语言控制和 forward pass，验证 tokenizer 与模型之间的维度一致性和语义正确性。产出的本地 HF checkpoint 供 TD-10 CTranslate2 转换。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- TD-08 产物目录中的 MVP 默认 tokenizer
- README MVP 模型配置表

## 执行事项

- 编写最小验证脚本 `scripts/verify_integration.py`。
- 用 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 加载 MVP 默认 tokenizer，立即断言 `tokenizer.is_fast is True`。
- 验证 encoder 输入构造：`<src_lang> source_text </s>` 的 tokenize 结果正确（token 序列包含 language token 首 token 和 `</s>` 尾 token）。
- 验证 decoder 端：`forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)` 可指定目标语言，`labels` 格式为 `<tgt_lang> target_text </s>`。
- 用最小 `M2M100Config` 创建一个随机初始化模型实例（使用 MVP 默认候选对应的配置，如 `e8-d2-v32k`）。
- 在模型初始化前硬断言 `len(tokenizer) == config.vocab_size`。
- 初始化后硬断言：`model.config.vocab_size`、`model.model.encoder.embed_tokens.num_embeddings`、`model.model.decoder.embed_tokens.num_embeddings`、`model.lm_head.out_features` 均等于 `len(tokenizer)`。禁止依赖后续隐式截断、补行或 `resize_token_embeddings()` 修复。
- 跑通一次 forward pass（随机 batch），确认无维度错误。
- 验证 encoder 输入的 attention_mask 遮蔽 padding 正确；decoder 的 causal mask 随 `forced_bos_token_id` 正确工作。
- 验证 encode/decode 往返正确性：四语样例编码后解码不丢失关键内容。
- 将微型随机模型和同一 tokenizer 保存到一个本地 HF checkpoint 目录（`artifacts/tokenizers/integration-checkpoint/`）。

## 产物

- `scripts/verify_integration.py`
- `artifacts/tokenizers/integration-checkpoint/`（本地 HF 目录，供 TD-10）
- 集成验证运行记录

## 验收

- `len(tokenizer) == config.vocab_size` 硬断言通过。
- encoder/decoder embedding 和 `lm_head` 行数均等于 `len(tokenizer)`。
- 一次 forward pass 无维度错误。
- 四语 encode/decode 往返正确。
- 本地 HF checkpoint 目录完整，可被 `AutoModelForSeq2SeqLM.from_pretrained()` 重新加载。
- 脚本可在无网络条件下运行。

## 验证记录

（待填写）

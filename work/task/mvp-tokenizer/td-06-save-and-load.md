# task TD-06: 产物保存与 AutoTokenizer 加载验证

状态：pending

依赖：TD-03（训练脚本与候选 tokenizer）、TD-04（tokenizer 构造规范）

## 目标

验证 `save_pretrained()` 产物的完整性和 `AutoTokenizer.from_pretrained()` 的离线加载一致性，确认规范 `tokenizer.json` 的 BPE 类型和 token ID 在保存前后一致。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- TD-03 训练产出的 32k 和 48k tokenizer（保存前）
- TD-04 语言 token 映射和 special token ID 基线

## 执行事项

- 调用 `tokenizer.save_pretrained(artifact_dir)` 保存完整 tokenizer 目录。
- 确认保存文件至少包含规范 `tokenizer.json`、`tokenizer_config.json` 和必要的 special token 配置；不得生成"文件名是 BPE、内容是 Unigram"的伪兼容文件。
- 验证 `AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True)` 返回锁定版本统一后的 `NllbTokenizer`，且 `tokenizer.is_fast is True`；不依赖 `facebook/nllb-200-*` 等远端仓库。
- 解析保存后的 `tokenizer.json`，验证 `model.type == "BPE"`，并对比保存前后完整 `get_vocab()` 和 backend 管线配置（Metaspace 参数、decoder、post-processor）。
- 通过 Python `tokenizers.Tokenizer.from_file(tokenizer_json)` 直接加载规范文件，与 `AutoTokenizer` 比较完整 ID→token 映射以及固定样例在不添加 special token 时的编码和 decode 结果。
- 验证语言 token encode/decode 行为：`eng_Latn` 编码为单一 token，decode 后仍为 `eng_Latn`。
- 验证 `(<src_lang> source_text </s>, <tgt_lang> target_text </s>)` 格式的 tokenize 结果符合预期。
- 验证 32k 和 48k 候选中最终 `len(tokenizer)` 与 `M2M100Config.vocab_size` 一致。
- 验证保存前后 `get_vocab()` 完整一致（所有 token 的 ID 相同）。

## 产物

- 32k 和 48k 的完整 `tokenizer` 保存目录（`artifacts/tokenizers/mvp-32k/`、`artifacts/tokenizers/mvp-48k/`）
- 保存前后一致性验证记录

## 验收

- `tokenizer.json` 中 `model.type == "BPE"`。
- `AutoTokenizer` 离线重载成功且 `is_fast is True`。
- `tokenizers.Tokenizer.from_file()` 直接加载与 `AutoTokenizer` 的 token→ID 映射一致。
- 语言 token 编码为单一 ID，decode 返回原始 token 字符串。
- 保存前后 `get_vocab()` 无差异。
- 最终 `len(tokenizer)` 精确为 32k 或 48k，与 M2M100Config 一致。
- 产物目录不含 `sentencepiece.bpe.model` 或其他 Unigram 伪装文件。

## 验证记录

（待填写）

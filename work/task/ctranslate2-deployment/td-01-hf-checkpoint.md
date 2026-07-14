# task TD-01: 微型 HF checkpoint

状态：pending

依赖：无

## 目标

基于冻结的 `mvp-tokenizer-v0` 生成可重复、可离线重载、可供 CTranslate2 converter 消费的随机微型 `M2M100ForConditionalGeneration` checkpoint。

## 输入

- [CTranslate2 deployment todo](../../todo/ctranslate2-deployment.md)
- `artifacts/tokenizers/mvp-tokenizer-v0/`
- 项目锁定的 Transformers、PyTorch 和 CTranslate2 版本

## 执行事项

- 从本地加载 tokenizer 并验证冻结 manifest、49,152 词表、fast backend 和五语言 allowlist。
- 固定随机种子，使用足以完成 forward/转换的最小 M2M100 encoder-decoder 配置。
- 在模型构造前后验证 `config.vocab_size`、shared/encoder/decoder embedding 和 `lm_head` 行数。
- 保存 checkpoint 后用 `AutoModelForSeq2SeqLM.from_pretrained(..., local_files_only=True)` 离线重载。
- 记录生成配置、版本、随机种子、文件列表、大小和 SHA-256。

## 验收

- 同一生成配置可重复构建语义等价的 checkpoint。
- 保存和离线重载成功，tokenizer/模型 ID 空间均为 49,152。
- checkpoint 不依赖远端仓库或第三方 tokenizer 文件。
- 文档明确随机 checkpoint 仅用于部署接口验证。

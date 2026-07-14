# task TD-01: 微型 HF checkpoint

状态：done

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

## 实现记录（2026-07-14）

- 固定配置：[micro_m2m100_deployment.json](../../../../configs/micro_m2m100_deployment.json)，随机种子 `20260714`，`d_model=32`，单层 encoder/decoder，词表严格绑定为 49,152。
- 生成与校验脚本：[build_micro_m2m100_checkpoint.py](../../../../scripts/build_micro_m2m100_checkpoint.py)。脚本先验证冻结 tokenizer manifest，再生成 Safetensors checkpoint，复制同一 tokenizer 供本地 converter 使用，并以 `local_files_only=True` 离线重载模型与 tokenizer。
- 可再生 checkpoint 位于 `artifacts/ctranslate2/runtime/hf-micro-checkpoint/`，由 Git 忽略；机器可读记录见合并 [deployment-validation.json](../../../../artifacts/ctranslate2/deployment-validation.json) 的 `phases.td_01_hf_checkpoint`。
- 重载后的 shared/encoder/decoder embedding 与 `lm_head` 均为 49,152 行，输入/输出 embedding 保持 tied；随机模型参数量 1,594,368。
- state dict SHA-256 为 `c2c82c641eb0b57f89cd077461042b4df87866ac192564ac08c386547c65ed07`，checkpoint manifest SHA-256 为 `0293d738ac1a063981ec42ddcc6474f594330e7205046452a376d2923d3e7351`；冻结 tokenizer manifest SHA-256 仍为 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。
- 复现命令：`.\.conda\python.exe scripts\build_micro_m2m100_checkpoint.py --spec configs/micro_m2m100_deployment.json --output-dir artifacts/ctranslate2/runtime/hf-micro-checkpoint --overwrite`。
- 验收环境：Python `3.11.15`、PyTorch `2.13.0+cu132`、Transformers `5.13.1`、Tokenizers `0.22.2`、Safetensors `0.8.0`、CTranslate2 `4.8.1`。
- 自动化测试 [test_micro_m2m100_checkpoint.py](../../../../tests/test_micro_m2m100_checkpoint.py) 覆盖双次生成完全相同、离线重载、有限 logits、产物文件、拒绝意外覆盖、合并报告和 Windows 瞬时文件锁重试；完整套件 `52 passed`。
- 随机 checkpoint 仅验证部署接口和 ID 空间，不具备也不代表任何翻译质量。

本 task 已随整个 todo 通过统一 review 并归档。

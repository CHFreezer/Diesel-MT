# task TD-05: 离线部署包与自动化测试

状态：done

依赖：TD-04

## 目标

固化 tokenizer 与 CT2 模型分离的部署布局，在新离线进程中复现 CPU 冒烟，并将关键边界纳入自动化慢速集成测试。

## 输入

- [CTranslate2 deployment todo](../../todo/ctranslate2-deployment.md)
- TD-01 至 TD-04 的已验证产物、报告和日志

## 执行事项

- 定义部署根目录，至少包含独立 `tokenizer/`、`model/`、manifest 和运行说明。
- 从干净临时目录启动新进程，禁用网络并仅使用部署根目录完成 tokenize、translate、去 prefix 和 decode。
- 增加自动化慢速测试，覆盖 checkpoint 生成、转换、逐 ID 词表比较、五语言 prefix 与 int8 CPU 冒烟。
- 区分可提交的生成逻辑/精简日志与不提交的大体积随机权重/转换缓存。
- 汇总版本、命令、哈希、运行时间、已知限制和失败排查方式。

## 验收

- 离线新进程端到端冒烟通过，运行时不访问远端资源。
- 部署 manifest 可验证全部必需文件，缺失或篡改时明确失败。
- 慢速集成测试可重复运行并通过。
- 部署说明和验收记录明确本工作流只验证部署兼容性，不代表翻译质量或生产性能。

## 实现记录（2026-07-14）

- 生成 Git 忽略的 `artifacts/ctranslate2/runtime/offline-package-int8/`，其中 `tokenizer/` 与 `model/` 独立，根 manifest SHA-256 为 `a19a189fb1288cec8d2d232601086b4b1fb0dc4a80902cd7a0f478ce301c5c86`。
- 自包含 runner：[run_offline_ctranslate2_smoke.py](../../../../scripts/run_offline_ctranslate2_smoke.py)。发布包复制该 runner，并在推理前逐文件验证部署 manifest。
- 新子进程从干净临时工作目录启动，移除 `PYTHONPATH`，启用 Hugging Face 离线标志、dead local proxy 和 Python socket connect guard；只从部署根加载 tokenizer/model，INT8 CPU 冒烟实际 compute type 为 `int8_float32`。
- 慢速集成测试：[test_ctranslate2_deployment.py](../../../../tests/test_ctranslate2_deployment.py)，覆盖 checkpoint、双次转换确定性、49,152 项逐 ID 词表、两种模型五语 prefix/decode、离线新进程、路径穿越/不完整 manifest 和文件篡改失败。完整套件为 `52 passed`。
- 复现命令：`.\.conda\python.exe scripts\validate_ctranslate2_deployment.py --phase package --overwrite`；机器可读文件哈希、运行时间与环境版本见合并 [deployment-validation.json](../../../../artifacts/ctranslate2/deployment-validation.json) 的 `phases.td_05_offline_package`，部署说明见 [artifacts/ctranslate2/README.md](../../../../artifacts/ctranslate2/README.md)。
- 已知限制：随机权重不提供翻译质量结论；冒烟不是生产延迟/吞吐基准；部署包要求兼容的 Python、Transformers 和 CTranslate2 runtime。
- 排错顺序：manifest 不匹配时重建包；模型加载失败时核对记录的 CTranslate2 版本与 CPU compute type；prefix/decode 失败时回查 TD-03 的有序词表与特殊 token 哈希。
- 随机模型只证明部署接口兼容，不代表翻译质量、量化精度或生产性能；本 task 未单独创建 review，最终随 todo 通过统一 review。

本 task 已随整个 todo 通过统一 review 并归档。

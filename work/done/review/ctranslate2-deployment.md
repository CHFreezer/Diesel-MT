# review: CTranslate2 deployment validation

状态：passed

复核日期：2026-07-14

## 范围

本 review 在 TD-01 至 TD-05 全部完成后统一执行，覆盖可重复性、转换正确性、完整 49,152 ID 空间、特殊 token、五语言 CPU 接口、离线与网络边界、manifest 防篡改、自动化测试、Git 边界和工作流记录。不评价随机模型的翻译质量、量化精度或生产性能。

## 证据

- plan：[CTranslate2 deployment validation](../../plan/ctranslate2-deployment.md)
- todo：[CTranslate2 deployment validation](../todo/ctranslate2-deployment.md)
- task 集合：[CTranslate2 deployment tasks](../task/ctranslate2-deployment/index.md)
- 部署说明：[CTranslate2 artifacts](../../../artifacts/ctranslate2/README.md)
- 合并机器记录：[deployment-validation.json](../../../artifacts/ctranslate2/deployment-validation.json)
- 实现：[checkpoint builder](../../../scripts/build_micro_m2m100_checkpoint.py)、[deployment validator](../../../scripts/validate_ctranslate2_deployment.py)、[offline runner](../../../scripts/run_offline_ctranslate2_smoke.py)
- 测试：[micro checkpoint](../../../tests/test_micro_m2m100_checkpoint.py)、[CT2 deployment](../../../tests/test_ctranslate2_deployment.py)

## 复核结果

- [x] 固定种子连续构建得到相同 state dict、payload 文件和 checkpoint manifest；离线模型/tokenizer 重载与有限 logits forward 通过。
- [x] float32 与 int8 转换均从本地 checkpoint 完成；CPU 实际 compute type 分别为 `float32` 和 `int8_float32`。
- [x] 冻结 tokenizer、HF checkpoint、float32 CT2 和 int8 CT2 的 49,152 项词表逐 ID 完全一致；有序词表 SHA-256 为 `72bc2edcfe44bdfac90d2c101f71f214cdd2c4b70d7c975e5d269011be40c716`。
- [x] `<s>`、`<pad>`、`</s>`、`<unk>`、五个 language token、source 边界、`target_prefix`、去 prefix 和 decode 合约全部通过。
- [x] float32/int8 各覆盖五个 source 和五个 target language；随机 hypothesis 未被用于语义质量结论。
- [x] 离线包分离 `tokenizer/` 与 `model/`，新进程从干净临时目录启动，移除 `PYTHONPATH`，启用 HF 离线标志、dead proxy 和 Python socket guard，只从部署根加载文件。
- [x] manifest 验证逐文件大小/SHA-256、精确文件集、`status=complete`、规范相对路径，并拒绝路径穿越、符号链接、不完整状态和文件篡改。
- [x] 可再生 checkpoint、转换目录和离线包均在 `artifacts/ctranslate2/runtime/` 下且未被 Git 跟踪；冻结 tokenizer 目录无改动。
- [x] 最终 `.conda\python.exe scripts\validate_ctranslate2_deployment.py --phase all --overwrite` 成功，五阶段结果合并为单一 JSON，完整回归为 `52 passed`，无残留 failure log。

## Review 中发现并修复

1. manifest 校验原先未在读取前明确拒绝路径穿越/符号链接，也未统一要求完成状态；已收紧校验并增加失败测试。
2. Windows 上报告文件可能被扫描器短暂锁定，单次 `os.replace` 曾触发 `PermissionError`；已为原子文件替换加入仅针对该异常的有限指数退避重试及自动化测试。
3. TD-01/TD-05 的历史测试数量记录已同步为最终 `52 passed`；各阶段 report Markdown 已合并进对应 task 和本统一 review，机器记录收敛为一个 JSON。

## 关键哈希

- 冻结 tokenizer manifest：`eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`
- 随机 HF state dict：`c2c82c641eb0b57f89cd077461042b4df87866ac192564ac08c386547c65ed07`
- HF checkpoint manifest：`0293d738ac1a063981ec42ddcc6474f594330e7205046452a376d2923d3e7351`
- float32 conversion manifest：`8f1a3c372efb74198ff2978cc157f36f57b3e18b6626a756e0bcf24e27041dee`
- int8 conversion manifest：`2e20a9ec780500cba0e7ec155f0f195a29ef9624b526dbc8c898d5ca33bf3512`
- offline package manifest：`a19a189fb1288cec8d2d232601086b4b1fb0dc4a80902cd7a0f478ce301c5c86`

## 结论

通过。该工作流已经证明冻结 tokenizer 与从零初始化 M2M100 checkpoint 的 CTranslate2 CPU 部署接口闭环兼容，可以整体归档。结论仅限部署兼容性；下一阶段若开展正式模型训练、翻译质量或性能工作，必须创建新的 plan/todo。

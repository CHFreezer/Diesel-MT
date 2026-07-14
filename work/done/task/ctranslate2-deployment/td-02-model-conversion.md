# task TD-02: CTranslate2 转换

状态：done

依赖：TD-01

## 目标

使用锁定版本的 CTranslate2 Transformers converter，将本地微型 M2M100 checkpoint 转换为 float32 诊断产物和 CPU int8 验收产物。

## 输入

- [CTranslate2 deployment todo](../../todo/ctranslate2-deployment.md)
- TD-01 生成并校验的本地 HF checkpoint

## 执行事项

- 记录 converter 版本、完整命令、环境变量、输入 checkpoint 哈希和输出目录。
- 先转换 float32 基线，再转换 int8；失败时保留可诊断日志，不发布半成品 manifest。
- 使用 CTranslate2 Python API 在 CPU 上分别加载两个产物，记录实际 compute type。
- 校验转换目录必需文件并生成大小/SHA-256 manifest。
- 确认转换期间无远端下载、`trust_remote_code` 或浮动模型引用。

## 验收

- float32 与 int8 均转换成功并可由 CPU runtime 加载。
- 输入输出 provenance、命令、版本、耗时和哈希齐全。
- 失败不会留下被误认为有效产物的完成标记或 manifest。

## 实现记录（2026-07-14）

- 统一执行器：[validate_ctranslate2_deployment.py](../../../../scripts/validate_ctranslate2_deployment.py)，阶段命令为 `.\.conda\python.exe scripts\validate_ctranslate2_deployment.py --phase convert --overwrite`。
- 输入 HF checkpoint manifest SHA-256 为 `0293d738ac1a063981ec42ddcc6474f594330e7205046452a376d2923d3e7351`，state dict SHA-256 为 `c2c82c641eb0b57f89cd077461042b4df87866ac192564ac08c386547c65ed07`。
- float32 转换 manifest SHA-256 为 `8f1a3c372efb74198ff2978cc157f36f57b3e18b6626a756e0bcf24e27041dee`，CPU 实际 compute type 为 `float32`。
- int8 转换 manifest SHA-256 为 `2e20a9ec780500cba0e7ec155f0f195a29ef9624b526dbc8c898d5ca33bf3512`，CPU 实际 compute type 为 `int8_float32`。
- 转换固定 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，`trust_remote_code=False`、`revision=None`。每种格式先在临时目录完成转换、必需文件检查、CPU 加载和 manifest，再原子发布；失败详情写入 Git 忽略的 `runtime/logs/last-failure.json`。
- 双次转换得到相同 payload 和 conversion manifest 哈希。机器可读记录见合并 [deployment-validation.json](../../../../artifacts/ctranslate2/deployment-validation.json) 的 `phases.td_02_conversion`。

本 task 已随整个 todo 通过统一 review 并归档。

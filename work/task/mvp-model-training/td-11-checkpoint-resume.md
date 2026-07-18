# task TD-11: 实现原子 checkpoint 与精确恢复

状态：completed

依赖：TD-10

## 目标

实现包含完整训练状态、身份校验和故障安全发布的 checkpoint/resume，使同一锁定环境中的中断恢复保持连续训练语义。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-10 训练循环、采样器状态与运行日志 schema
- TD-01 staging/发布和配置身份契约

## 原子边界

本 task 只实现和验证 checkpoint 机制；不执行正式 M1/M2 长训练，不决定 checkpoint 的质量优劣，也不清理任何未被新完整 checkpoint 替代的产物。

## 执行事项

- 保存模型、optimizer、scheduler、scaler、global step、epoch、已消费样本/token、累积相位、采样器及 Python/NumPy/PyTorch CPU/CUDA RNG 状态。
- manifest 绑定数据/config/tokenizer/code/依赖哈希、Git commit/dirty、设备/CUDA、文件大小/SHA-256 和 `status=complete`。
- 使用同目录 staging、逐文件 fsync/校验与最终原子发布；拒绝不完整、缺失、哈希错误、路径穿越、符号链接和身份错配。
- 实现显式 `--resume-from`，确保恢复后不重放/跳过样本，不重置 scheduler、累积相位或 RNG。
- 注入写权重、optimizer、manifest 前后故障，证明半成品不发布且旧 checkpoint 保持可用。
- 比较 uninterrupted/resumed 短训练的 step、学习率、采样、loss 和权重；优先精确相等，已证实非确定算子须预先冻结容差。
- 定义保留/清理策略；仅在新 checkpoint 完整验证后允许删除旧产物。

## 产物

- checkpoint/resume 模块和完整性验证器。
- 故障注入、连续/恢复一致性测试与报告。
- checkpoint manifest 和保留策略。

## 验收

- 任一 complete checkpoint 可恢复连续训练语义。
- 损坏、错配或不完整 checkpoint 均在恢复前失败。
- 故障不会覆盖最后一个可用 checkpoint 或发布半成品。
- 恢复一致性满足冻结的精确值或有证据的容差。

## 实现与运行证据

2026-07-16 完成 TD-11：

- 新增 `scripts/mvp_checkpoint.py`，checkpoint 固定保存 model、parameter gradients、optimizer、scheduler、scaler、trainer/sampler 状态以及 Python/NumPy/PyTorch CPU/CUDA RNG；trainer state 包含 global/micro step、epoch、已消费 sample/token、梯度累积相位、loss history、逐路由曝光和 token audit。
- manifest 绑定 training/student/tokenizer/data、四个训练代码文件、依赖、Git commit/dirty、设备/精度/Torch/CUDA 身份；每个 payload 记录 byte count 与 SHA-256，`status=complete` 且 identity hash 验证成功后才允许恢复。
- checkpoint 在目标根的同目录 staging 写入，每个 payload 与 manifest 均 flush + `fsync`，逐文件哈希复验后使用目录 `os.replace` 原子发布；目标已存在时拒绝覆盖。恢复前拒绝 staging/incomplete、文件缺失/额外、byte/hash 损坏、路径穿越、symlink/reparse 和 identity mismatch。
- 训练入口提供显式 `--checkpoint-root`、`--resume-from` 和受控 `--stop-after-optimizer-step`；现统一为 `scripts/mvp_cli.py train`。checkpoint callback 只在 optimizer 边界发布；恢复不会重置 sampler、scheduler、累积相位或 RNG，也不会重复/跳过样本。
- 四个故障注入点 `after_model`、`after_optimizer`、`before_manifest`、`after_manifest_before_publish` 均证明半成品不会发布且旧 complete checkpoint 字节不变。保留策略默认 keep-last 3、禁止自动清理；只有最新 checkpoint 再验证通过后才可显式清理旧 complete 目录。
- 当时由 `scripts/validate_mvp_checkpoint_resume.py` 生成的验收现迁移为 `scripts/mvp_cli.py validate-resume`，并由 `mvp_training.validate_resume_equivalence` 复用：正式 student 连续 2 step 与 step 1 中断/恢复到 step 2 的 loss、optimizer/micro step、sampler state、完整语义 trace 和最终七个 payload 哈希全部精确一致；模型/optimizer/trainer payload SHA-256 分别为 `6a129d3fff098cdddb0c86f9f7dd5cfa2144f79ff1da3681ddb6b0ab9dc50390`、`27c3726a5e8afefd76768754e86eedac8c291c7c4dbe02759c57d72f658ee32f`、`40eb52f9d5d80183f5056895814068233e9c8c65c9a2e000fe9ecdd2b4ddad3f`。
- 最终 checkpoint identity SHA-256 为 `7b27cc5e567bf3b245e97a18925d518fcbbbff44a26fa2d37aeb26795af75aac`；连续/恢复 semantic trace SHA-256 均为 `b937866624470c1764aacaab155690826eebb0f841d11159d1d83b0ef1236b74`。manifest 因创建时间不同而不同，所有训练 payload 完全相同。
- 机器可读证据 `artifacts/model-training/reports/student/checkpoint-resume.json` SHA-256 为 `8c32d2a700e13bcc08e468e4312a3d3a48ae5e1c134d1add82af41b28338efb8`；定向测试 `.conda\python.exe -m pytest tests/test_mvp_training.py tests/test_mvp_checkpoint.py -q` 为 `15 passed, 1 skipped`。跳过项仅是当前 Windows 权限不允许测试创建 symlink；实现仍以 symlink 与 reparse attribute 双重拒绝，普通路径穿越/额外文件/损坏测试均已执行。

TD-11 只关闭 checkpoint/resume 机制；正式 M1 过拟合、生成记忆和最终 HF checkpoint 由 TD-12 验收。

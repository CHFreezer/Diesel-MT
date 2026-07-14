# task TD-11: 实现原子 checkpoint 与精确恢复

状态：pending

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

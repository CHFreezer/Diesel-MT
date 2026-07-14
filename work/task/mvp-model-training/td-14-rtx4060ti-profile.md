# task TD-14: 基准测试并冻结 RTX 4060 Ti 训练配置

状态：pending

依赖：TD-05、TD-12

## 目标

在 RTX 4060 Ti 16 GB 和真实长度分布上找到唯一、稳定、可恢复且保留显存安全余量的 M2 student 训练 profile。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 train/dev 小切片及长度分布
- TD-12 通过 M1 的正式 student、训练与 checkpoint 链
- 本机 CUDA/PyTorch、D: NVMe staging 与 E: 数据盘边界

## 原子边界

本 task 只做 student 容量/稳定性基准并冻结 profile，不比较 human/distilled 质量、不执行完整 M2 预算，也不修改数据或 tokenizer。

## 执行事项

- 验证本机 BF16 与锁定 CUDA/PyTorch 稳定性；任何精度或环境回退都记录理由和新身份。
- 用真实长度分布切片比较 micro batch、梯度累积、gradient checkpointing、最大长度和 dataloader worker。
- 记录每个候选的峰值显存、吞吐、step/验证耗时、OOM/重试和截断率，禁止只用短 synthetic 样本估算。
- 选择唯一 M2 profile，冻结 optimizer/scheduler、batch、累积、长度、验证/checkpoint 频率和 step/token 预算。
- 验证 E: 机械盘不成为 checkpoint 热路径，SSD staging 与最终发布遵守 TD-01 边界。
- 完成至少 100 optimizer step 的 soak，包含至少 2 次 dev 验证和 2 次 checkpoint 发布，无显存持续增长、NaN/Inf、停顿或写入阻塞。

## 产物

- 硬件候选基准、资源曲线和选择报告。
- 唯一冻结的 RTX 4060 Ti M2 profile 与配置哈希。
- 100-step soak、恢复和 checkpoint 发布记录。

## 验收

- 唯一 profile 在 16 GB 内保持安全余量并通过 soak。
- 截断率、吞吐和验证/checkpoint 开销满足预设门槛。
- 热写入位于 SSD，发布路径和身份完整可追溯。
- TD-16 只能消费该冻结 profile，不得临时调参。

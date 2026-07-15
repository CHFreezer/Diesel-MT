# task TD-14: 基准测试并冻结可配置训练资源 profile

状态：pending

依赖：TD-05、TD-12

## 目标

在当前执行主机和真实长度分布上，基于显式资源预算找到唯一、稳定、可恢复且保留安全余量的 M2 student 训练 profile。实现不得按 GPU 型号或固定显存容量分支；换机时调整资源 profile，不修改训练算法。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 train/dev 小切片及长度分布
- TD-12 通过 M1 的正式 student、训练与 checkpoint 链
- `configs/mvp_e8_d2_v48k.yaml` 的 provisional resource schema
- 根目录 Git-excluded `LOCAL_HARDWARE.md`（仅作本机规划参考；以运行时探测为准）

## 原子边界

本 task 只做 student 容量/稳定性基准并冻结 profile，不比较 human/distilled 质量、不执行完整 M2 预算，也不修改数据或 tokenizer。

## 执行事项

- 启动时探测 accelerator 数量/型号、后端与驱动、支持精度、设备总内存和主机可用内存，并写入 benchmark/run manifest；语义配置不保存 GPU 型号。
- 从配置读取设备/精度候选和 `device_memory_budget_mib`、`device_memory_reserve_mib`、`max_device_memory_utilization`、`host_memory_budget_mib`、`dataloader_memory_budget_mib`、`oom_retry_limit`；任何回退都记录理由和新 profile 身份。
- 用真实长度分布切片比较 micro batch、梯度累积、gradient checkpointing、最大长度和 dataloader worker；候选组合全部来自配置，禁止隐藏常量。
- 记录每个候选的峰值设备/主机内存、吞吐、step/验证耗时、OOM/重试和截断率，禁止只用短 synthetic 样本估算。
- 选择唯一 M2 profile，冻结资源预算、optimizer/scheduler、batch、累积、长度、验证/checkpoint 频率和 step/token 预算。
- 验证 checkpoint/staging/log 热路径使用配置的高吞吐运行根，最终发布遵守 TD-01 边界，不依赖盘符。
- 完成至少 100 optimizer step 的 soak，包含至少 2 次 dev 验证和 2 次 checkpoint 发布，无显存持续增长、NaN/Inf、停顿或写入阻塞。

## 产物

- 硬件候选基准、资源曲线和选择报告。
- 唯一冻结的 M2 训练资源 profile、配置哈希和运行时硬件 manifest。
- 100-step soak、恢复和 checkpoint 发布记录。

## 验收

- 唯一 profile 的实测峰值不超过配置预算，并满足预留内存/利用率约束和 soak 门槛。
- 截断率、吞吐和验证/checkpoint 开销满足预设门槛。
- 训练代码和自动化测试中不存在 GPU 型号或固定显存容量分支；修改资源 profile 即可适配另一容量等级的设备。
- 热写入位于配置的运行根，发布路径和身份完整可追溯。
- TD-16 只能消费该冻结 profile，不得临时调参。

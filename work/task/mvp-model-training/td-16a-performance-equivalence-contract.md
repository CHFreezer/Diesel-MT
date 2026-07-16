# task TD-16A: 合并性能优先且硬件可配置的训练器

状态：completed（2026-07-17）

依赖：TD-14、TD-15、TD-16 A/B 诊断

## 目标

把隔离 worktree 中验证过的高吞吐训练管线合并到主分支；训练结果以能力和 time-to-quality 负责，不再要求权重、逐步 loss 或 CUDA 热路径逐 bit 一致。

## 原子边界

本 task 只定版通用训练实现、资源配置接口、恢复语义和测试，不提交本机专用 batch、worker、显存、盘符或运行根，也不启动新的完整质量训练或消费正式 test。

## 已完成实现

- 离线文本/token 编码缓存、可选持久缓存和确定性长度分桶；恢复时保存并校验 sampler pending state。
- 可配置的预编码 worker、pinned memory、non-blocking 传输、micro batch、梯度累积、显存/主存预算和 checkpoint 保留数。
- 可配置的 fused AdamW、CUDA allocator、梯度有限性检查策略和日志模式/刷新频率。
- 运行时探测 CPU、RAM、GPU、VRAM 和后端能力；配置超过实际逻辑 CPU 或内存预算时 fail-fast。
- 数据、tokenizer、配置和 complete checkpoint 继续做完整性校验；训练热路径不以逐 batch hash 阻塞吞吐。

## 验收证据

- 主分支提交：`f842b4a`（`训练：合并可配置的高吞吐训练管线`）。
- 本机吞吐候选 YAML 与正式运行根保持 Git-ignored，没有进入提交。
- 完整离线测试：`204 passed`。
- 重复训练验收保留“能力等价而非权重 hash 相等”的原则；具体质量阈值必须在 TD-02～TD-05 v3 冻结新 corpus/dev 后注册，不能沿用已证明不适合作为通用 MT 验收集的旧阈值。

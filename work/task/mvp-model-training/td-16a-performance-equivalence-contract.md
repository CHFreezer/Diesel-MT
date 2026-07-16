# task TD-16A: 定版性能优先训练器与能力等价合同

状态：pending（已有隔离候选与基准，不等于主分支验收完成）

依赖：TD-14、TD-15、TD-16 A/B 诊断

## 目标

将完整 M0 上验证过的训练管线优化整理为可配置、可测试、可恢复的正式训练入口，并冻结“重复训练能力等价、权重 hash 不必相同”的验收合同。

## 原子边界

本 task 只定版训练实现、资源 profile、time-to-quality 指标和能力等价门槛；不启动完整 human 质量训练，不消费正式 test，不决定蒸馏混合比例。

## 执行事项

- 审查并集成离线 token cache、长度分桶、pinned/non-blocking 传输、`cudaMallocAsync`、fused AdamW、performance 日志和有限值同步优化。
- 在完整 226,218 条 human train 选择范围上完成有界吞吐/显存/功耗基准，保留长训显存安全余量；不以 GPU 100% 或满 TDP 代替端到端吞吐。
- 将 hash 限制在数据/tokenizer/config/checkpoint 完整性边界；训练热路径不计算逐 batch 精确语义 hash。
- 保留 NaN/Inf、OOM、零截断、路由曝光、checkpoint 可加载和恢复后训练语义连续性检查。
- 冻结能力等价指标、重复次数、总体及逐路由容差、最大墙钟/训练 token 预算和 early-stop 规则；判定依据只来自 human dev。
- 用短程重复运行验证配置可重放、loss 有限和性能稳定；不要求模型权重逐 bit 相同。

## 产物与验收

- 正式性能优先训练配置、实现、测试和完整 M0 候选报告。
- time-to-quality 与能力等价合同在任何完整训练结果产生前冻结。
- 全套离线测试通过，TD-16/正式 test 旧证据未被覆盖。

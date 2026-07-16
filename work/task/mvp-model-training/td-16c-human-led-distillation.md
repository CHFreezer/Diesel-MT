# task TD-16C: 执行 human 主导的蒸馏辅助训练

状态：pending

依赖：TD-16B

## 目标

从最佳 human foundation 继续训练，验证低比例 Hy-MT2 sequence target 能否以更短墙钟或相同墙钟提高 human dev 能力；纯 distilled 不作为默认续训方案。

## 原子边界

本 task 只比较在 TD-16B checkpoint 上继续 human-only 与预先冻结的 human-led teacher 辅助配方。不访问 test，不新增 teacher 数据，不无条件把44,313条相同 source 的human/teacher记录各训练一次。

## 执行事项

- 在查看本阶段结果前冻结 teacher target 的候选比例/课程、较低学习率、最大 token/墙钟预算、评测频率和停止门槛。
- 同一 source 每次曝光只选择一个 target；记录 human/teacher target 实际曝光，不因存在双 target 自动将 source 权重翻倍。
- 以继续 human-only 作为等墙钟或等 token 对照，比较达到相同 dev 门槛所需时间及最终总体/逐路由能力。
- 若 teacher 辅助未提升 time-to-quality、触发任一路由退化或整体不如继续 human-only，保留负结果并选择 human-only。
- 发布 human-only continuation 与最佳 human-led candidate 的 dev-only 证据。

## 产物与验收

- 预先冻结的混合/课程配置、曝光报告、等预算对照和候选 checkpoint。
- 任何 teacher 候选都必须同时满足总体增益、逐路由退化红线和脚本/空输出/source-copy 门槛；否则不得进入 TD-16D。

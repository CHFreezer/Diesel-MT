# task TD-16D: 验证 human-led teacher 辅助

状态：pending

依赖：TD-16C，以及与新 M0 source 对齐的 teacher 数据门槛

## 目标

从 TD-16C 的最佳 human foundation 出发，验证低比例 Hy-MT2 sequence target 是否缩短 time-to-quality 或提高新 human dev 能力。

## 原子边界

本 task 不访问 test，不用旧 D1 自动替代新语料的 teacher 数据，不无条件把同一 source 的 human/teacher target 各训练一次。

## 执行事项

- 在查看结果前冻结 teacher target 身份、比例/课程、继续 human-only 对照、最大 token/墙钟预算、学习率和停止门槛。
- 同一 source 每次曝光只选择一个 target；分别记录 human/teacher 曝光和达到相同 dev 门槛所需时间。
- teacher 辅助必须同时通过总体增益、逐路由退化红线、实体/数字忠实度、脚本、空输出和 source-copy 门槛；否则选择 human-only。
- 只按 dev 发布 human-only continuation 与最佳 human-led candidate，不消费正式 test。

## 产物与验收

- human-only continuation、human-led teacher 辅助等预算对照、time-to-quality 报告和唯一候选配方。
- teacher 辅助不改善时如实保留负结果，TD-16E 使用 human-only 配方。

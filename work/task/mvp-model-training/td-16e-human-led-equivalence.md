# task TD-16E: 验证 human-led teacher 辅助与重复能力等价

状态：pending（等待 TD-16D human foundation）

依赖：TD-16D

## 目标

从 TD-16D 的最佳 human foundation 出发，验证低比例 Hy-MT2 sequence target 是否缩短 time-to-quality 或提高新 human dev 能力，并对最终配方做能力等价复跑后冻结唯一候选。

## 原子边界

本 task 不访问 test，不新增 teacher 数据，不要求模型 hash 相同，也不无条件把同一 source 的 human/teacher target 各训练一次。

## 执行事项

- 在查看结果前冻结 teacher target 比例/课程、继续 human-only 对照、最大 token/墙钟预算、学习率和停止门槛。
- 同一 source 每次曝光只选择一个 target；分别记录 human/teacher 曝光和达到相同 dev 门槛所需时间。
- teacher 辅助必须同时通过总体增益、逐路由退化红线、实体/数字忠实度、脚本、空输出和 source-copy 门槛；否则选择 human-only。
- 对最终配方按预注册次数独立复跑，以总体及20路 dev 能力容差和 time-to-quality 验证统计等价；允许权重、逐步 loss 和 checkpoint hash 不同。
- 只按 dev 冻结一个候选和不可变 test 授权记录；不能挑最好的一次掩盖不稳定，也不能为失败追加训练预算。

## 产物与验收

- human-only continuation、human-led teacher 辅助等预算对照、重复训练能力等价报告、time-to-quality 分布和唯一候选。
- 唯一候选满足冻结的总体与20路能力容差；正式 test 消耗仍为 `0/1`。

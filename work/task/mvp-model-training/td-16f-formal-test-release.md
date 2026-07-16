# task TD-16F: 执行一次性正式 test 并发布 MVP

状态：pending

依赖：TD-16E

## 目标

只对 TD-16E 冻结的唯一候选执行一次正式 test，发布第一个基于修复后 human 语料能力训练的 MVP HF 模型与质量报告。

## 原子边界

本 task 不训练、不调参、不重试 test、不比较多个候选。test 结果无论好坏都必须如实发布；失败不能通过再次消费 test 解决。

## 执行事项

- 验证唯一候选、授权 receipt、test 身份和 `runs_consumed=0`。
- 执行一次正式 test，原子写入 receipt 后禁止再次运行。
- 报告20个标签路由、12个跨语言产品方向和2个简繁互转操作的 loss、BLEU/chrF、实体/数字忠实度、脚本合规、空输出、source copy和目标控制。
- 发布可离线重载的 HF checkpoint、训练数据/预算/墙钟摘要、能力等价证据和已知弱项。
- 明确区分“本项目有界语料上的 MVP”与生产质量模型，不夸大结果。

## 产物与验收

- 一次性正式 test 报告、不可重放 receipt、最终 HF MVP manifest 和模型卡式能力摘要。
- test 只消费一次；TD-16A～TD-16F 全部完成后，TD-16 才可标记 `completed`，TD-17 方可开始。

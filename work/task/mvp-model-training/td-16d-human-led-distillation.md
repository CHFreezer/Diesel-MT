# task TD-16D: 执行或跳过一次弱路由数据补强

状态：pending

依赖：TD-16C

## 目标

只在 TD-16C 没达到 MVP 门槛且错误集中于明确路线/领域时，设计一次最小有界的数据补强并验证是否改善 time-to-quality；如果 human-first 基线已经过线，则记录 `not-needed` 并直接进入 TD-16E。

## 原子边界

本 task 不访问 test，不预设必须使用 Hy-MT2、DeepSeek 或固定 synthetic 比例，不把同一 source 的多个 target 无条件重复训练，也不以扩大模型或延长无上限训练代替数据诊断。

## 决策顺序

1. 先按 TD-16C 的逐路由生成、实体/数字/术语、脚本、source-copy 和领域错误定位缺口。
2. 优先回到 TD-02～TD-05 补真实 human parallel；只有真实来源不足且用户明确授权，才建立 source-only synthetic 生成和独立语义复审合同。
3. 在生成/补训前冻结新增 groups/tokens、来源或 teacher、prompt、费用、混合/课程、继续 human-only 对照、最大 token/墙钟预算和退化红线。
4. 同一 source 每次曝光只选择一个 target；human 与 synthetic 分账，dev/test 始终只用 human reference。

## 蒸馏的剩余价值与候选边界

蒸馏生成语料仍有价值，但只作为 human-first 基线后的定向补强，不再承担 60M 底模的主体语料。优先适用三类缺口：Hant–JA/Hant–KO 等真实平行关系稀缺、经许可和质量确认的近期单语文本带来的新实体/新术语、以及 TD-16C dev 已定位的实体/术语/语域等具体错误类型。

- 当前没有默认合格的生成 teacher：Hy-MT2 v3 已因系统性实体/术语错误被拒；DeepSeek 直译虽在 512 条 A/B 中更好，也没有获得全量生成授权。任何 teacher/API 都必须重新完成路线级校准、费用和质量合同。
- 只有继续寻找真实 human parallel 仍不能关闭弱路由，且用户另行授权后，才运行一次等训练预算 A/B：`human-only continuation` 对比 `human + bounded synthetic`。
- synthetic 的首个候选曝光档位为全局约 5%～10%，单个已证实弱路由不超过约 20%；这是试验搜索边界，不是必须用满的固定配额。不得用一条 source 的多个 target 重复曝光虚增训练量。
- 生成 target 必须独立复审并与 human 分账；选择只看 human-only dev 的总体、逐路由、实体/数字/术语、脚本和 time-to-quality。没有明确收益或造成其他路线退化时，保留负结果并回到 TD-16C 配方。

## 验收

- 补强必须改善预注册弱路由，同时不触发总体和其他路由退化红线；否则回退到 TD-16C human-first 配方。
- 输出 `not-needed`、`human-augmentation-selected`、`synthetic-augmentation-selected` 或 `no-improvement` 中唯一结论。
- 只冻结一个进入 TD-16E 的配方，不消费正式 test。

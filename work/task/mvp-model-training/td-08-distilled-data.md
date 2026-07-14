# task TD-08: 生成并验收有界 sequence-level 蒸馏数据

状态：pending

依赖：TD-05、TD-07

## 目标

使用锁定的 Hy-MT2 7B artifact 和唯一 prompt/decode profile，只对冻结 train source 生成覆盖 18 路由的有界离散 teacher targets，并完成过滤、人工抽检、重放和原子发布。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 M0 train source/group manifest
- TD-06 teacher artifact/runtime lock
- TD-07 语言映射、prompt/decode 和过滤阈值

## 原子边界

本 task 只做离线 sequence-level 数据生成：不保存 logits/hidden states，不让 teacher 进入 student 训练图，不读取 test，也不启动 student 训练。

## 执行事项

- 实现 `scripts/generate_teacher_data.py`，只接受冻结 train source/`sample_group_id`，显式拒绝 dev/test，并覆盖 18 路由。
- 支持 dry-run、确定性分片、原子 shard、逐样本 checkpoint/resume、缓存校验和中断恢复；worker/batch/resume 不得改变规范输出身份。
- 每条记录保存 teacher revision/hash、后端、prompt version、decode config/seed、输入 sample/group ID、raw response、normalized target 及哈希。
- 分开保存 raw response 与 accepted target；按冻结规则过滤空输出、解释/echo、source copy、错语言/脚本、异常长度、截断、重复和占位符损坏。
- 每路由至少人工检查 20 条 accepted 和 20 条 rejected，不足时全检；繁体额外检查简繁混淆、地区词和共享汉字误判。
- 报告 18 路由输入/成功/拒绝/重试、长度、脚本、来源和吞吐；任一路由低于门槛即停止发布。
- 对固定分片独立重放，验证 raw/normalized 输出和 manifest 符合 TD-07 复现契约。
- 原子发布有界 distilled train corpus 与 complete manifest；dev/test 继续只保留人类参考。

## 产物

- `scripts/generate_teacher_data.py` 与生成/恢复测试。
- Git-ignored raw/accepted/filtered teacher 数据。
- 18 路由质量、人工抽检、重放和完整 provenance 报告。
- D0 完成 manifest。

## 验收

- plan 的 D0 门槛全部满足，18 路由均有通过过滤的 teacher targets。
- teacher、prompt、decode、输入和输出身份可逐样本追溯。
- 固定分片重放通过，失败不会发布半成品。
- test 从未被读取；只有 accepted targets 可进入 TD-15。

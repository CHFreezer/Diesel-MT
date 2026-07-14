# task TD-05: 构建并验收 M0 数据集

状态：pending

依赖：TD-04

## 目标

运行正式有界数据构建，发布覆盖 5 标签、9 无向组和 18 有向路由的 M0 corpus/fixture，并以人工质量、泄漏和字节级复现证据关闭数据前置门槛。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-02 锁定来源与预算
- TD-03/TD-04 已验收的数据构建、split、去重和泄漏模块

## 原子边界

本 task 负责 M0 数据运行和发布，不生成 Hy-MT2 蒸馏 targets、不实现 student 编码，也不以扩大数据量修复来源或质量缺口。

## 执行事项

- 建立 `tests/fixtures/model_data/`，覆盖 9 组无向标签对、18 有向路由及非法路由反例。
- 构建有界真实 MVP corpus，确认 5 标签桶、9 无向组和 18 路由均非空，简体与繁体分别有独立 dev/test。
- 冻结方向采样策略，报告原始/过滤/正反扩展数、训练权重和有效曝光，禁止低资源方向无界重复。
- 逐标签对/split 报告来源占比、原生/synthetic/增强占比、长度分布、脚本合规和过滤原因。
- 每组至少人工检查 20 条 accepted train、10 条 accepted dev/test 和 20 条 rejected；不足时全检，并覆盖繁体与混合脚本边界。
- 以不同 worker/cache 状态完成两次独立构建，逐文件比较 corpus、manifest 和确定性报告 SHA-256。
- 发布 M0 验收报告；空路由、繁体 dev/test 缺失、泄漏、来源不明或复现失败均阻塞发布。

## 产物

- 有界 MVP corpus 与 18 路由 fixture。
- 覆盖、质量、泄漏、人工抽检和复现报告。
- `status=complete` 的 M0 manifest。

## 验收

- plan 的 M0 数据门槛全部满足。
- 两次独立构建的规范 corpus/manifest 字节级一致。
- 简体、繁体数据身份和 dev/test 均独立可追溯。
- 数据集明确标记为可供 TD-07、TD-09、TD-12～TD-16 消费。

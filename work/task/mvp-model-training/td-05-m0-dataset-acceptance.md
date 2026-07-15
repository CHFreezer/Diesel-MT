# task TD-05: 构建并验收 M0 数据集

状态：completed

依赖：TD-04

## 目标

保留已发布的 5 标签、9 组/18 路 M0 v1，补充 `zho_Hans--zho_Hant` 第 10 组并发布 20 路 human composite，以人工质量、泄漏和字节级复现证据重新关闭数据前置门槛。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-02 锁定来源与预算
- TD-03/TD-04 已验收的数据构建、split、去重和泄漏模块

## 原子边界

本 task 负责 M0 数据运行和发布，不生成 Hy-MT2 蒸馏 targets、不实现 student 编码，也不以扩大数据量修复来源或质量缺口。

## 执行事项

- 保留 9 组/18 路 v1 fixture/corpus；扩展 fixture 到 10 组/20 路，增加简繁互转、共享汉字、词汇差异和泄漏反例。
- 构建并验收第 10 组，确认两条新增路线及其独立 train/dev/test 非空，再发布引用 v1 与 addendum 的 human composite。
- 冻结方向采样策略，报告原始/过滤/正反扩展数、训练权重和有效曝光，禁止低资源方向无界重复。
- 逐标签对/split 报告来源占比、原生/synthetic/增强占比、长度分布、脚本合规和过滤原因。
- 每组至少人工检查 20 条 accepted train、10 条 accepted dev/test 和 20 条 rejected；不足时全检，并覆盖繁体与混合脚本边界。
- 以不同 worker/cache 状态完成两次独立构建，逐文件比较 corpus、manifest 和确定性报告 SHA-256。
- 发布 M0 验收报告；空路由、繁体 dev/test 缺失、泄漏、来源不明或复现失败均阻塞发布。

## 产物

- 不可变 M0 v1、第 10 组 human addendum、20 路 composite 与 fixture。
- 覆盖、质量、泄漏、人工抽检和复现报告。
- `status=complete` 的 M0 manifest。

## 验收

- plan 的 M0 数据门槛全部满足。
- 两次独立构建的规范 corpus/manifest 字节级一致。
- 简体、繁体数据身份和 dev/test 均独立可追溯。
- 只有 20 路 composite 明确标记为可供 TD-07、TD-09、TD-12～TD-16 完整验收消费；M0 v1 单独不再满足门槛。

## 实现与运行证据

- 原版 FLORES-200 仅作为外部评测污染引用：官方归档、仓库 revision、许可证和五标签 `dev/devtest` 已由 `configs/mvp_mt_evaluation.lock.json` 锁定；10,045 条 exact/near 扫描命中为 0，未进入训练 corpus。
- 正式 corpus 为 147,443 个无向关系、294,886 条有向样本；train/dev/test 分别为 203,942/33,490/57,454，5 标签、9 组、18 路由全部非空，简体/繁体分别具有独立 dev/test。
- `configs/mvp_direction_sampling.yaml` 固定 18 路由统一权重 1.0、每 epoch 最多一次曝光，禁止低资源无界重复；完整 pair/split 来源、provenance、长度、长度比、脚本与过滤统计写入运行时 TD-05 报告。
- 固定人工队列逐条审查 489 条：accepted 360、rejected 129、繁体混合脚本 9；accepted 标记 29 个已知截断/错位质量问题，rejected 规则不匹配为 0。该警告保留在提交内 attestation，不宣称生产翻译质量。
- 首次冷构建与第二次热缓存/完全离线 fresh 构建逐字节比较 10 个规范产物全部一致；第二输出根的五 locale resume checkpoint 也全部命中且 manifest 不变。
- runtime `m0-manifest.json` SHA-256 为 `3d517a8adad0871d04f688f8fd50e0e6432ea0738a043bb8977f1dcede3c37aa`，acceptance report SHA-256 为 `4b64f8f3562978332396c11f6708a2ba507442d230d1615a5a36a09f04a40b1a`；精简证据见 `artifacts/model-training/m0-dataset-acceptance.json` 与 `docs/m0-dataset-acceptance.md`。
- 专项测试 `20 passed`；完整离线回归 `105 passed in 22.89s`。冻结 tokenizer artifact manifest 仍为 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。

以上是 2026-07-15 的 M0 v1 完成记录。

## 20 路 M0 完成记录（2026-07-16）

- M0 v2 覆盖 10 组/20 路、327,508 条有向记录；train/dev/test 为 226,218/37,508/63,782。第 10 组最终无向 train/dev/test 为 11,279/1,868/3,164。
- 固定人工队列逐条检查 549 条：accepted 400、rejected 149；第 10 组 accepted train/dev/test 和 rejected 均达到冻结抽检预算。保留 38 个已知 MASSIVE 边界质量警告，无 systemic blocker。
- 双构建、污染、脚本、来源/许可、20 路非空与方向采样 gate 全部通过。M0 manifest SHA-256 为 `5cc369421a705e2eea0076eec06c2bc12de7f278888df2f1ca9add6250ee1d67`；提交内证据为 `artifacts/model-training/td05-m0-20route.json`。
- v1 artifact 与哈希保持不变；TD-09 未启动。

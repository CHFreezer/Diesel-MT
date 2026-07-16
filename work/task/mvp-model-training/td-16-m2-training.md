# task group TD-16: 训练并冻结基于现有语料能力的 MVP 模型

状态：in_progress（A/B 诊断已完成；完整 MVP 能力训练未完成）

依赖：TD-05、TD-08、TD-12、TD-13、TD-14、TD-15

## 目标澄清

TD-16 的最终含义必须是：使用已经收集并验收的 226,218 条 20 路 human train 语料作为能力基础，训练出经过独立 dev 选择、重复训练能力等价验收和一次性正式 test 的 `mvp_e8_d2_v48k` MVP 模型。

原 TD-16 把“44,313 条共同 source 上的 human-only/distilled 等预算 A/B”与“完成 MVP 模型训练”放在同一个原子任务中，导致 A/B 候选被误解为已经吸收完整语料能力的模型。这个边界不成立，现拆分为 TD-16A～TD-16E；只有五个原子任务全部完成，TD-16 才能标记 `completed`。

## 已完成事实：M2 A/B 诊断，不是完整 MVP 训练

2026-07-17 已完成原冻结合同中的两臂训练与 dev-only 选择：

- 两臂只消费 TD-15 的 44,313 条共同 source，约占完整 human train 记录数的 19.6%；每条 source 在 human-only 臂使用人类 target，在 distilled 臂使用 Hy-MT2 target。
- 两臂各有 1,000 optimizer-step 上限、有效 batch 128，属于 source-matched target A/B；它们不是 human 与 distilled 联合训练，也没有先训练完整 human 底模。
- human-only 最佳候选为 step 1,000；distilled 最佳候选为 step 900。冻结规则最终选择 human-only step 1,000。
- distilled 候选未通过总体 chrF 增益、SacreBLEU 退化和任一路由最大 chrF 退化门槛；这是“纯 teacher target 不能替代 human target”的负结果，不等于 teacher target 永远不能作为低比例辅助监督。
- 正式 test 未运行，额度仍为 `0/1`；TD-16 不能据此宣称已发布最终 MVP。

选择证据位于 `D:\Diesel-MT-Runtime\td16-m2-v1\selection.json`。该运行必须保留为不可变 A/B 证据，不得改写成完整语料训练结果。

## 不再成立的完成判定

以下任一事实都不足以完成 TD-16：

- random/M1 checkpoint 能保存、恢复或部署；
- 44,313 条共同 cohort 上任一 1,000-step A/B 候选被 dev 选中；
- 完整 M0 上完成 100-step 吞吐或资源 soak；
- loss 下降但未验证 20 路生成能力；
- 单次训练成功但没有重复训练能力等价证据；
- 模型权重或逐步 trace hash 一致。

## 原子拆分

1. [TD-16A：定版性能优先训练器与能力等价合同](td-16a-performance-equivalence-contract.md)
2. [TD-16B：训练完整 human M0 底模](td-16b-full-human-foundation.md)
3. [TD-16C：执行 human 主导的蒸馏辅助训练](td-16c-human-led-distillation.md)
4. [TD-16D：验证重复训练能力等价并冻结唯一候选](td-16d-capability-equivalence-selection.md)
5. [TD-16E：执行一次性正式 test 并发布 MVP](td-16e-formal-test-release.md)

TD-16A～TD-16E 严格串行。TD-17 CTranslate2 回接必须等待 TD-16E 完成。

## 总验收

- 完整 human M0 的 226,218 条记录全部进入训练抽样范围，20 路均有可审计曝光；不得用44,313条 A/B cohort 冒充完整语料。
- 最终配方以最短 time-to-quality 为目标；允许 BF16、fused optimizer、异步 allocator、SDPA、缓存、预取、长度分桶和非 bitwise CUDA 路径。
- 重复训练不要求模型 hash、逐步 loss 或权重逐 bit 相同；能力等价由冻结 human dev 的总体与20路 BLEU/chrF、loss、脚本合规、空输出、source copy和目标控制容差判定。
- teacher target 只能按预先冻结的低比例/课程配方作为辅助候选；不得把相同 source 的 human/teacher target 无条件双倍拼接，也不得在看到 dev 结果后追加预算。
- 唯一最终候选冻结前不访问 test；正式 test 只执行一次。
- TD-16E 发布的 checkpoint 能离线重载，且文档明确其数据规模、训练预算、能力指标和已知弱项。

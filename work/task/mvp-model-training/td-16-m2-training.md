# task group TD-16: 训练并冻结基于合格语料能力的 MVP 模型

状态：pending / suspended（TD-16A、TD-16B 诊断已完成；当前活动入口已回退到 TD-02）

恢复依赖：TD-05 v3；teacher 辅助阶段还必须重新验证 TD-08/TD-15 与新 source identity 的兼容性

## 目标与边界修正

TD-16 的最终含义是：以经过通用翻译忠实度验收的 20 路 human 语料为能力基础，训练出经过独立 dev 选择、重复训练能力等价验收和一次性正式 test 的 `mvp_e8_d2_v48k` MVP 模型。

原 44,313 条共同 source A/B 只回答“teacher target 能否替代 human target”。随后完整 226,218-record M0 长训又证明：记录数和 complete manifest 不能代替独立语义规模与翻译忠实度。失效前提位于 TD-02 的来源选择，所以当前工作已离开 TD-16，重新执行 TD-02～TD-05；A/B 和长训证据均保留，但不能被重新解释为最终 MVP。

## 已完成事实

### A/B 诊断

- human-only/distilled 两臂各训练 1,000 optimizer steps，只消费 TD-15 的44,313条共同 source，正式 test 未访问。
- 冻结 dev 规则选择 human-only step 1,000；纯 teacher target 未通过总体与逐路由门槛。
- 该结果不代表完整 human foundation，也不代表 human-led teacher 辅助一定无效。

### 训练器合并

- 主分支已经合并可配置的缓存、预编码/分桶、pinned/non-blocking 传输、fused optimizer、allocator、日志和恢复路径。
- 硬件型号、worker、batch、RAM/VRAM 数值和运行根不写死在代码中；由 YAML、运行时探测和 Git-ignored 本机 profile 决定。
- 完整离线测试为 `204 passed`。

### 完整 M0 长训否决

- 226,218 条 directed records 只有 11,411 个 semantic/alignment groups；平均每组扩展为 `19.82` 条路由记录，不能按 226,218 个独立语义理解。
- 训练 validation loss 在 step 4,000 后持续回升，按授权在 step 15,000 checkpoint 后 early-stop；5k/10k/15k 均完成 dev-only 诊断，正式 test 未运行。
- MASSIVE 是窄领域多语言本地化。当前 `utt` 直接对齐允许地点、媒体、服务和人物的 locale substitution；这对 intent/localization 有效，但不满足通用机器翻译的 source fidelity。
- 现有 checkpoint 只保留为训练器/数据诊断，不进入蒸馏、复跑、test 或发布。

## 当前回退路径（不属于 TD-16 子任务）

1. TD-02 v3：重新调研并锁定通用 MT 平行来源 — in_progress
2. TD-03 v3：适配新来源并构建新 corpus identity — pending
3. TD-04 v3：重新切分、去重与泄漏/污染审计 — pending
4. TD-05 v3：重新验收并发布新 M0 — pending

TD-05 v3 完成以前，TD-16 不处于执行状态，也不得启动新 foundation、teacher 辅助或正式 test。

## 未来恢复路径

1. [TD-16A：合并性能优先且硬件可配置的训练器](td-16a-performance-equivalence-contract.md) — completed
2. [TD-16B：验证完整 human M0 长训并定位语料边界](td-16b-full-human-foundation.md) — completed，候选不准入
3. [TD-16C：训练新 M0 human foundation](td-16c-repaired-human-foundation.md) — pending，等待 TD-05 v3
4. [TD-16D：验证 human-led teacher 辅助](td-16d-human-led-distillation.md) — pending
5. [TD-16E：验证重复训练能力等价并冻结候选](td-16e-capability-equivalence-selection.md) — pending
6. [TD-16F：执行一次性正式 test 并发布 MVP](td-16f-formal-test-release.md) — pending

重新进入 TD-16 后，TD-16C～TD-16F 严格串行。TD-17 CTranslate2 回接必须等待 TD-16F 完成。

## 总验收

- 新 human corpus 同时报告 directed records、唯一 language/text 和 independent semantic groups；不得用路由展开冒充语义规模。
- 训练 target 必须满足通用 MT 的语义、实体、数字和否定忠实度；localization substitution 只能进入显式标注的辅助用途。
- 最终配方以最短 time-to-quality 为目标；允许 BF16、fused optimizer、异步 allocator、SDPA、缓存、预取、长度分桶和非 bitwise CUDA 路径。
- 重复训练不要求模型 hash、逐步 loss 或权重逐 bit 相同；能力等价由新冻结 human dev 的总体与20路指标及错误门槛判定。
- 唯一最终候选冻结前不访问 test；正式 test 只执行一次。
- TD-16F 发布的 checkpoint 能离线重载，且明确数据规模、独立语义规模、训练预算、能力指标和已知弱项。

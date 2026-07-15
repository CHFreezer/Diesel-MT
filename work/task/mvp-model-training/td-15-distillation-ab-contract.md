# task TD-15: 冻结蒸馏配方与等预算 A/B 契约

状态：pending

依赖：TD-05、TD-08、TD-13

## 目标

建立 source 身份完全相同、只在训练 target/provenance 上不同的 human-only 与 distilled 两组 recipe，并在查看 M2 结果前冻结预算和 dev 选择规则。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 human train references 与 route/group manifest
- TD-08 D1 accepted teacher targets 与 provenance manifest；D0 smoke 不具备正式 A/B 输入资格
- TD-13 固定评测协议

## 原子边界

本 task 只构建和验证 A/B cohort/recipe，不执行正式 M2 训练，不用 test 决定混合、阈值或候选，也不把 rejected teacher target 回退到单独一组。若输入是 D0 smoke、任一路由 D1 accepted 少于 2,000 或 D1 总 accepted 少于 36,000，必须在 cohort 构建前拒绝。

## 执行事项

- 只以 D1 accepted teacher targets 与 human references 的交集建立固定 cohort；D0 smoke 禁止进入正式 recipe，teacher 失败/rejected source/group 必须从两组同时排除。
- human-only 使用人类 target，distilled 对完全相同 source/group ID 使用 Hy-MT2 target；两组 dev/test 都只用冻结人类参考。
- 冻结相同的 student 初始 state-dict hash、source 顺序、路由权重、batch/累积、optimizer/scheduler、最大 step 和 eval/checkpoint 频率。
- 定义等预算为相同 source 曝光序列与 optimizer step；target 长度差异独立报告，禁止看到结果后追加 step/样本/方向曝光。
- 训练前逐路由比较样本数、source/target token、截断率、脚本合规和 target 差异，验证 18 路由 source 身份及曝光逐项一致。
- 预先冻结 dev 选择优先级/阈值，覆盖 chrF/SacreBLEU、loss、脚本合规、空输出/source-copy 和逐路由最大退化，繁简分开判定。
- 发布两份不可变 recipe/manifest 和差异报告；除 target/provenance/hash 外的优化字段差异必须导致校验失败。
- 分别完成短 dry-run，验证采样、初始权重、step 边界和评测入口一致，且无法访问 test。

## 产物

- 共同 A/B cohort manifest。
- human-only/distilled 两份冻结 recipe 与配置哈希。
- 等预算校验器、训练前差异报告和双 dry-run 证据。

## 验收

- 两组 source/group、初始化、曝光和优化预算完全相同。
- 预期差异只限于训练 target 及其 provenance/hash。
- dev 选择规则在 M2 结果出现前冻结，test 保持隔离。
- 任一公平性校验失败都会阻塞 TD-16。
- D1 数量/身份门槛失败或误用 D0 smoke 同样阻塞 TD-16。

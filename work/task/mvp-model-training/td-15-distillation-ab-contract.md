# task TD-15: 冻结蒸馏配方与等预算 A/B 契约

状态：completed

依赖：TD-05、TD-08、TD-13

## 目标

建立 source 身份完全相同、只在训练 target/provenance 上不同的 human-only 与 distilled 两组 recipe，并在查看 M2 结果前冻结预算和 dev 选择规则。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 human train references 与 route/group manifest
- TD-08 的 20 路 distilled composite 与 provenance manifest；D0/D1 v1 单独不具备正式 A/B 输入资格
- TD-13 固定评测协议

## 原子边界

本 task 只构建和验证 A/B cohort/recipe，不执行正式 M2 训练，不用 test 决定混合、阈值或候选，也不把 rejected teacher target 回退到单独一组。若输入不是 20 路 composite、任一路由 accepted 少于 2,000，或错误使用 D0/D1 v1 单体，必须在 cohort 构建前拒绝。

## 执行事项

- 只以 20 路 distilled composite 与 human composite 的 accepted 交集建立固定 cohort；teacher 失败/rejected source/group 必须从两组同时排除。
- human-only 使用人类 target，distilled 对完全相同 source/group ID 使用 Hy-MT2 target；两组 dev/test 都只用冻结人类参考。
- 冻结相同的 student 初始 state-dict hash、source 顺序、路由权重、batch/累积、optimizer/scheduler、最大 step 和 eval/checkpoint 频率。
- 定义等预算为相同 source 曝光序列与 optimizer step；target 长度差异独立报告，禁止看到结果后追加 step/样本/方向曝光。
- 训练前逐路由比较样本数、source/target token、截断率、脚本合规和 target 差异，验证 20 路 source 身份及曝光逐项一致。
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
- composite 数量/身份门槛失败或误用 D0/D1 v1 单体同样阻塞 TD-16。

## 完成记录

- 冻结 `configs/mvp_distillation_ab.yaml`，只允许 TD-08 `hymt2-sequence-distillation-d1-20route-composite-v2` train-only composite；D0/D1 v1 或单独中文 addendum、非 20 路、任一路由少于 2,000、非 teacher-synthetic provenance 均在 cohort 构建前拒绝。目录规范化后的契约文件 SHA-256 为 `e5a3ed538fe7c1306fabaedd0f7931609e6dbf8457c98a946f6a93ee60f3f50b`。
- 严格按 `(sample_group_id, src_lang, tgt_lang, normalized source)` 取 human train 与 44,361 条 accepted teacher composite 的交集，发布 44,313 条共同 cohort；20 路最少 2,207。48 条 accepted teacher 记录因当前 human train 无相同身份而从两臂共同排除；其余 rejected/filtered teacher source 也不会进入任一组，`teacher_rejected_or_filtered_included=0`。
- 共同 source order SHA-256 为 `a05e3131224e25cbf97e59f14b96f3f016ef132db7936ae2610e08c0c93a8c6c`。human/distilled corpus SHA-256 分别为 `b848fc057f6e76ddc021bfc8687bc68283f255bfebb4441fbdab45d0b8b70a05`、`a17bf12890abfaf7147b2341cf552e9ab0edb71d0d50507ecbe0e9b20e24ad8f`；共同 cohort 为 `6cb36b07d75fc11b7b89acb5bc98786d6b9889ea22164c5c71ca06ce31f79bae`，manifest 为 `576369e021ccd9fb45437e7b4671c046acb8945e9b4fbdd63379841d499865c5`。
- 训练前统计显示 human/distilled target 分别 468,047/509,385 tokens，target difference rate `0.979035497483808`；两臂 source 与 target 截断均为 0，整体脚本合规分别 `0.999232730801345`/`0.999819466070905`。target 长度差异只报告，不补样本或 step。
- 冻结可直接供 TD-16 使用的 `configs/mvp_training_m2_human.yaml` 与 `configs/mvp_training_m2_distilled.yaml`，文件 SHA-256 分别为 `2c209c731153aad861460744adea29218ce2c9ae0a9e7db9aeb8d287df56c293`、`1929032d5f3ed6f3ef012b814b603f688f0b75ec619738dac65046f5a284a267`。两者绑定相同 initial state `66897f9c358802b9d39d66e61a8b39fad21236d11744b79df194c26db4da66a3`、TD-14 profile、TD-13 evaluator、route weights、1,000 optimizer step/5,000,000 token 上限及每 50 step dev/checkpoint。
- 512 次 paired dry-run 的 source sequence SHA-256 为 `29fb392abf735f94a7f487639e43f2e3fdf3990505963401a2e020f0003cd177`，两臂 selection/sampler state 精确一致；两份正式训练 config 的独立 dry-run 均通过。dev 冻结为 human reference，选择顺序、逐路由退化阈值、`zho_Hans`/`zho_Hant` 分开门槛及失败回退 human-only 已在 M2 结果前冻结；TD-15 未访问 test。机器记录为 `artifacts/model-training/reports/m2/distillation-ab.json`。

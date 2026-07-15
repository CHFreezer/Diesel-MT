# task TD-08: 生成 D0 smoke 并验收 D1 最小可用蒸馏数据

状态：completed

依赖：TD-05、TD-07

## 目标

保留覆盖 18 个跨语言路由的 D0/D1 v1，使用同一锁定 Hy-MT2 7B artifact 和既有 `Chinese` / `Traditional Chinese` prompt/decode，只对新增两条简繁互转 train source 生成 addendum，最终发布 20 路 distilled composite。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 M0 train source/group manifest
- TD-06 teacher artifact/runtime lock
- TD-07 语言映射、prompt/decode 和过滤阈值

## 原子边界

本 task 只做离线 sequence-level 数据生成：不保存 logits/hidden states，不让 teacher 进入 student 训练图，不读取 test，也不启动 student 训练。D0 complete 只表示真实数据冒烟完成；D1 v1 的 18 路门槛已经通过，但只有新增两路各达到至少 2,000 accepted 并发布 20 路 composite，才允许重新关闭本 task 并供 TD-15 消费。

## 执行事项

- 保留覆盖 18 路 v1 的 `scripts/generate_teacher_data.py` 行为，并版本化支持两条新增路线；始终只接受冻结 train source/`sample_group_id`，显式拒绝 dev/test。
- 支持 dry-run、确定性分片、原子 shard、逐样本 checkpoint/resume、缓存校验和中断恢复；worker/batch/resume 不得改变规范输出身份。
- 每条记录保存 teacher revision/hash、后端、prompt version、decode config/seed、输入 sample/group ID、raw response、normalized target 及哈希。
- 分开保存 raw response 与 accepted target；按冻结规则过滤空输出、解释/echo、source copy、错语言/脚本、异常长度、截断、重复和占位符损坏。
- 每路由至少人工检查 20 条 accepted 和 20 条 rejected，不足时全检；繁体额外检查简繁混淆、地区词和共享汉字误判。
- 保留 18 路 v1 报告；新增两路分别报告输入/成功/拒绝/重试、长度、脚本、来源和吞吐，任一路由低于门槛即停止发布。
- 对固定分片独立重放，验证 raw/normalized 输出和 manifest 符合 TD-07 复现契约。
- 原子发布有界 distilled train corpus 与 complete manifest；dev/test 继续只保留人类参考。
- 冻结 D1 独立身份，沿用 D0 teacher/prompt/decode/filter，确定性选择 D0 source 的超集；每路由 2,224 个候选、总计 40,032，D1 不覆盖或改名复用 D0 artifact。
- 生成、过滤并验收 D1：每路由 accepted 至少 2,000、总 accepted 至少 36,000，并独立完成逐路由人工抽检、繁体/共享汉字专项审查、精确 replay 和 manifest-last 发布。
- 为 `zho_Hans -> zho_Hant` 与反方向各生成 2,224 个候选、至少接受 2,000 个；source-copy 采用路线专用规则，允许合法不变文本。
- 独立完成人工审查、精确 replay 和 manifest-last addendum，再发布引用 D1 v1 与新增两路的 20 路 composite；不得覆盖 v1。

## 产物

- `scripts/generate_teacher_data.py` 与生成/恢复测试。
- Git-ignored raw/accepted/filtered teacher 数据。
- 18 路 v1 与两路 addendum 的质量、人工抽检、重放和完整 provenance 报告。
- D0 smoke 完成 manifest（已存在，保留不变）。
- D1 mvp 独立 raw/accepted/filtered、质量报告、人工审查、replay 与完成 manifest（已完成）。

## 验收

- D0/D1 v1 证据保持有效，两条新增路线各有至少 2,000 条 accepted，20 路 composite 通过完整门槛。
- teacher、prompt、decode、输入和输出身份可逐样本追溯。
- 固定分片重放通过，失败不会发布半成品。
- test 从未被读取；只有 20 路 composite accepted targets 可进入 TD-15，D0 与 D1 v1 单独均不得替代 composite。

## D0 smoke 实现与运行证据

- 2026-07-15 完成 `configs/hymt2_distillation.yaml`、`scripts/hymt2_distillation_data.py` 和 `scripts/generate_teacher_data.py`；正式生成契约 SHA-256 为 `42bb80d67d428c40031ee880a86a74420f842388bdba8aa3ed25837c5c7a5fd0`。
- 只从冻结 train 输入按 18 路由各确定性抽取 128 条，共生成 2,304 条；墙钟时间 1,100.460623 秒、completion token 30,613、请求内 completion 吞吐 28.965 tokens/s。逐样本 checkpoint 写入 `DIESEL_MT_DISTILLATION_WORK_ROOT` 指定的热工作目录，raw/accepted/filtered 与 manifest 位于 Git-ignored 数据目录。
- 人工逐条检查 381 条队列：每路由 20 条 accepted，全部 6 条自动 rejected，以及 3 个繁体目标路由各 5 条额外样本。人工剔除 39 条语义错误；对 `話題`、`愛`、`岸田文雄`、`主旨` 4 条日中共享汉字的 `source_copy` 误杀执行受限人工恢复，其余硬过滤不可覆盖。审查证明冻结在 `configs/hymt2_distillation_manual_review.yaml`。
- 最终接受 2,263 条、过滤 41 条；18 路由最低接受率 0.960938，脚本合规率全部 1.0，重试率全部 0，质量门槛失败项为空。
- 使用同一 artifact/profile 独立重放每路由 2 条，共 36 条；raw 与 normalized 输出均精确一致，零 mismatch。生成和重放均记录 `dev_accessed=false`、`test_accessed=false`。
- complete manifest SHA-256 为 `2e0beb51e0b5020f7248da4d0f7bdd544bb0274c29c0efc22affa9d83ff1639e`，accepted SHA-256 为 `282be328032877cb9a380e76bb2b006e64822b0863825978e9cd8a5ee8bd2b81`；重复真实 replay 保持报告 SHA-256 `8cc4512ebf5d7f8a11567863104ad03648c82e78626727db76df9db82ae1c8a2`，连续两次 finalize 的 manifest/evidence/accepted/quality 哈希逐项一致。Git 跟踪的闭环证据为 `artifacts/model-training/td08-distilled-data.json`。
- 定向验证命令：`.conda\python.exe -m pytest tests\test_hymt2_distillation.py tests\test_hymt2_distillation_data.py -q`，结果 `22 passed`。TD-09 未启动。
- 最终全量离线回归：`.conda\python.exe -m pytest -q`，结果 `139 passed in 22.90s`。

## D1 mvp 实现与运行证据

- 2026-07-15 冻结 `configs/hymt2_distillation_d1.yaml`，D1 generation contract SHA-256 为 `2e54be92d270af3acac76251f25e31987a876f3e098dfb7bbbc73c696a470b1a`。选样沿用 D0 seed，使 D0 每路由 128 条成为 D1 每路由 2,224 条的精确前缀；运行前逐字节校验 D0 config、generation contract、complete manifest 和 18 个 raw shard，再把 2,304 条输出值重新绑定 D1 contract，未改动 D0 artifact。
- 18 路由共生成 40,032 条候选，其中新 teacher 推理 37,728 条；墙钟时间 18,382.615340 秒。逐样本 checkpoint 位于 `DIESEL_MT_DISTILLATION_D1_WORK_ROOT` 配置的热工作目录，18 个完整 raw shard 逐路由原子发布。
- 人工逐条检查独立 D1 队列 444 条：每路由 20 条自动 accepted、全部 69 条自动 rejected、3 个繁体目标路由各 5 条额外样本。人工剔除 52 条自动接受的语义、实体、数字、命令意图或脚本质量错误；只对 31 条有效共享汉字、数字、缩写和专名的 `source_copy` 单原因误杀执行受限恢复。审查证明为 `configs/hymt2_distillation_d1_manual_review.yaml`。
- 最终接受 39,941 条、过滤 91 条；18 路由 accepted 为 2,211～2,223，最低接受率 0.994155、最低脚本合规率 0.999101、重试率全部为 0，质量门槛失败项为空。总 accepted 和每路由 accepted 均显著高于冻结 D1 门槛。
- 独立重载同一 GGUF Q8_0 + llama.cpp CUDA runtime，每路由 replay 2 条，共 36 条；raw 与 normalized 输出逐字节一致，零 mismatch。generation、replay 和 evidence 均记录 `dev_accessed=false`、`test_accessed=false`。
- D1 v1 complete manifest SHA-256 为 `9de9a4c251504c9ee157bec2dc4eefea8acd760d808672c15704f5c884b9ff2c`，accepted SHA-256 为 `9602480b643954dbd030d4d1b768d140742d25d892d41da1393a99a2fd79dd57`，tracked evidence 为 `artifacts/model-training/td08-d1-distilled-data.json`。它曾满足 18 路输入门槛；范围修正后必须由 20 路 composite 引用才能进入 TD-15。
- 重复 finalize 后 manifest、tracked evidence、accepted、filtered、质量报告、人工审查队列和 replay 报告七类产物哈希全部保持一致。蒸馏专项回归结果为 `26 passed`，全量离线回归结果为 `143 passed in 87.49s`。

## 成熟度回退决定与最终收口

- 2026-07-15 复核发现 D0 仅占 M0 human train source 的约 1.11%，每路由只有 123～128 条 accepted；它足以证明 teacher 数据链正确，但不足以支撑 `mvp_e8_d2_v48k` 的正式训练或 human-only/distilled A/B 结论。
- D0 artifact、哈希和审查证据继续作为 immutable smoke 记录保留，不回写为 D1，也不删除成功证据。
- TD-08 曾因 D0 规模不足退回并完成 18 路 D1 v1；D1 v1 的 40,032 个候选、39,941 条 accepted 及全部哈希保持不可变。

## 20 路 D1 完成记录（2026-07-16）

- 从 20 路 M0 train 为两条中文内部路线各确定性选择 2,224 个候选；实际生成 4,448 条，墙钟 1,583.418523 秒，重试率均为 0。
- `zho_Hans->zho_Hant` 接受 2,213、过滤 11；`zho_Hant->zho_Hans` 接受 2,207、过滤 17。两路接受率与脚本合规率均通过冻结 gate，失败项为空。
- 人工逐条审查 72 条队列，剔除 3 条大陆词汇残留，恢复 2 条与 human reference 精确一致的合法不变繁体句。4 条独立 replay 的 raw/normalized 输出完全一致；dev/test 均未读取。
- addendum accepted/manifest SHA-256 分别为 `b9ca2fd5a05d9c9874548b7ea5a3db5dc75b5512438b46dac9ae65ca9e88fcb1` / `8700222adb328a4f7aac3dc92c46b53183dba7d1c46c97fd12e4d6eaab7a942f`。
- 最终 composite 只引用不可变 D1 v1 与两路 addendum，包含 44,361 条 accepted、20 路，每路至少 2,207 条；accepted/manifest SHA-256 为 `30178717aa24cf9a14a80db6cf9ed236469c7032b25fe82811999d2e09317604` / `fe72be6a588fda2a328e8c300d799061cab62ecfaabf13a702e637eb4dd8cd1e`。重复构建三类哈希全部一致，TD-09 未启动。
- 收口专项测试为 `33 passed`，全量离线回归为 `148 passed in 28.76s`。

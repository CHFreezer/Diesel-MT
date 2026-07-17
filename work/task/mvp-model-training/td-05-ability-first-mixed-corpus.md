# task TD-05 schema v4：发布并验收 80/20 ability-first mixed corpus

状态：pending runtime publication（实现和测试已完成；等待 TD-04 complete + manual review）

依赖：TD-03 schema v4、TD-04 schema v4

## 目标

将 TD-04 accepted teacher targets 与受控一跳 reverse pairs 组成 teacher pool，将 TD-03 的 40,000 条 human anchors 组成 human pool，发布一个不复制记录凑 raw 数量、但在训练曝光时严格按 teacher/human=`80/20` 采样的 20 路 train corpus。

## 冻结输入与身份

- 发布合同：`configs/mvp_60m_mixed_corpus.yaml`，规范 SHA-256 `d7dfceabe6d09d3a3e1e886fd29669deac91156a3d0cd96c980dd8cb635180c2`。
- TD-03 source bank、human anchors 与 runtime manifest 必须逐文件匹配已提交紧凑 manifest。
- TD-04 runtime manifest 必须 `status=complete`，generation config SHA-256 必须为 `d062b22cb853247460051fa131b40d61f5069ce5f93948175db1b1e31a1b847e`，accepted/reverse 文件必须逐字节匹配。
- 冻结 tokenizer artifact manifest SHA-256：`eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。

## 原子边界

本 task 只发布 train corpus 和采样合同，不训练 student，不创建 checkpoint，不读取 FLORES devtest/正式 test，不启动 TD-16。

## 发布与采样合同

- raw corpus 只包含实际 accepted teacher、合法 reverse 和 40,000 条 human anchors；`duplicate fill` 永久禁止。
- 每条记录保留 `sample_id`、`sample_group_id`、route、source/target、mixture class、原始 record/job ID、generation identity 和 provenance。
- teacher/human semantic groups 必须完全不相交；sample ID 和规范 directed text pair 必须唯一。
- teacher pool 必须覆盖全部 20 路；source 非 Hant 的固定 teacher 路线仍需至少 10,000 条。
- 训练曝光采用确定性两级 smooth weighted round-robin：第一级 class 精确 4:1，第二级在该 class 可用 routes 中均匀，再在 route 内确定性 shuffle/epoch cycling。100,000 次冻结 preview 必须恰为 80,000 teacher / 20,000 human。
- 80/20 是 sampling weight，不是 raw record 比例；不会复制原生繁体或 human 记录伪造规模。

## 质量门

- 使用冻结 tokenizer 对所有 source/target 无截断编码；任一侧超过 256 token 即阻止整个发布，不静默 truncate。
- 20 路、唯一 ID/pair、teacher/human group 隔离、输入 hash、manifest-last、formal-test isolation 均为 fail-fast 门。
- `scripts/validate_mvp_60m_data_chain.py` 最终串联验证 TD-02 source byte lock、TD-03、TD-04/manual review 和 TD-05；任一局部证据缺失都不能把数据链标记 complete。

## 产物与完成条件

- runtime：`td05/training-corpus.jsonl`、`td05/sampling-plan.json`、最后发布的 `td05/manifest.json`。
- tracked evidence：`artifacts/model-training/reports/m0/mvp-60m-td05-manifest.json` 与 `mvp-60m-data-chain.json`。
- `status=complete`、20 路、零截断、80/20 preview、无 raw duplicate fill、无 teacher/human group overlap、人工 review 无 blocker、完整回归通过后，才可关闭 TD-05。

# task TD-05: 发布并验收 human-first MVP 平行语料

状态：pending

依赖：TD-03、TD-04

## 目标

将通过确定性硬门、DeepSeek 长上下文辅助审计和人工校准的真实平行句对发布为首个 human-first MVP 训练 corpus。该发布是 TD-16C 唯一可消费的数据身份，不包含 Hy-MT2 v3 accepted、旧 D1 自动继承或 DeepSeek 自动修正文本。

## 发布规则

- 只消费 TD-03 `preaudit` 与 TD-04 audit overlay；被确认错误、未完成批次、低置信高风险和未关闭许可项全部隔离。
- 未被 DeepSeek 标记只表示本轮未发现问题；最终准入还须通过来源/路线/年份/领域分层的人工作为总体质量估计。
- 保留原始 human source-target，不为了补齐方向将同一句重复计数；正反训练展开共享同一 semantic group，并在统计中区分 groups、pairs、directed records 和 tokens。
- dev/test 只使用独立 human reference，按 group/document/work 隔离；训练、DeepSeek 审计调参与 checkpoint 选择不能访问正式 test。
- 报告 20 路覆盖、10 组关系、年份、领域、简繁、来源和许可证分布；近期内容与现代术语层按 TD-02B 冻结区间执行，不为了比例收低质量文本。
- Hant 继续质量实收，技术和法律/政务 ceiling 保留；港澳正式书面繁体可补充，粤语/广东话继续排除。
- 随正式 split 冻结一份未来 tokenizer 候选账本：只登记 train 分区内通过许可与质量门的唯一真人两侧文本，保留稳定 `text_id`、标签、来源、文档、年代、领域、许可和内容哈希；该账本不能被当前训练器消费，也不表示 64k 已准入或已训练。

## 决策边界

- 如果实收语义组、tokens、路线覆盖或人工估计忠实度未达到 TD-02B 冻结门槛，状态为 `blocked` 并回到 TD-02A/TD-02B 扩展来源；不得自动启用 Hy-MT2/DeepSeek 重译填洞。
- 如果 human-first corpus 足以启动能力训练，则发布唯一 manifest 并解锁 TD-16C。
- 任何 synthetic augmentation 都必须在 human-first 基线训练和 dev 诊断后另立有界任务，不能混入本 task 的完成定义。

## 产物

- train/dev/test 规范语料、source/license/attribution 清单和完整 manifest；
- audit overlay、quarantine 与人工决定的绑定摘要；
- groups/pairs/directed records/tokens、路线/年份/领域和质量统计；
- 数据卡，明确非商业研究用途、原始数据再分发边界和已知风险。
- 未来 64k tokenizer 候选账本及五标签唯一文本/字符量汇总；正文是否可本地保留或再分发继续服从逐来源许可。

## 完成条件

- 数据身份、许可、日期、硬门、DeepSeek batch、人工校准、split 和去重全部可追溯。
- 20 路训练覆盖与五标签 dev 覆盖满足冻结门槛，简体/繁体分别报告；正式 test 保持未消费。
- 训练入口拒绝任何非本 manifest 的旧 M0/D1/Hy-MT2 v3 或 preaudit 数据。
- tokenizer 候选账本排除 dev/test、tokenizer holdout、synthetic、canary、quarantine 和反向展开重复；未来任务仍须补充更广的高质量单语来源并重新做配比、去重、污染与饱和度验收。
- 发布后才能重写并启动 TD-16C human-first 60M 能力训练。

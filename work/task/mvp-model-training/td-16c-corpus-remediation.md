# task TD-16C: 修复并重新冻结 human 平行语料

状态：in_progress（2026-07-17 回退入口）

依赖：TD-16B

## 目标

在不覆盖现有 M0/D1/A/B/长训证据的前提下，重新定义并构建适合通用机器翻译底模的 human train/dev corpus。当前任务入口是语料准备，不再继续训练、蒸馏或正式 test。

## 原子边界

本 task 只做来源调研、许可/lock、语料适配、过滤、切分、统计和有界数据 pilot。不会重建冻结 tokenizer，不修改已发布 M0 v1/composite，不启动完整 foundation 长训，也不访问正式 test 指标。

## 执行事项

- 把“directed record 数”“唯一 `(language, text)` 数”“semantic/alignment group 数”分开报告；路由展开不得再被描述为等量新增语义。
- 逐来源声明 `literal_parallel`、`localization_parallel`、`teacher_synthetic` 等用途。MASSIVE 默认降级为窄领域/路由控制补充，不再作为通用 MT 主体。
- 审计 MASSIVE 的 `utt`、`annot_utt`、slot 和 locale adaptation：只有语义与实体都可忠实对应的记录才能进入 literal MT；若使用统一 slot placeholder，必须证明所有语言可逆且训练/推理语义一致。
- 为 5 个标签、10 组关系、20 路补充具有足够独立语义多样性的真实平行来源；锁定版本、文件 hash、许可、用途和逐路由数量。
- 重新执行 group-aware split、近重复/反向泄漏和外部评测污染检查。dev 必须独立于训练模板和本地化生成机制，并能检验实体、数字、否定、命名实体及目标脚本忠实度。
- 冻结 source/target fidelity 人工抽检门槛和逐来源拒绝原因；不能再把 locale entity substitution 自动视为通用翻译的正确 target。
- 用小规模 pilot 验证 tokenizer 截断、路由平衡、loss 有限和 dev 方向正确；pilot 只验证新 corpus 身份，不据此发布模型。

## 产物与验收

- 新的配置/source lock、不可变 corpus manifest、逐来源用途矩阵、唯一语义统计、质量/许可/泄漏报告和人工抽检证据。
- 20 路都有可审计覆盖，但能力规模按独立 semantic groups 和 token 数衡量，不以同一 multiparallel group 的路由展开充数。
- 新 dev 能检测本地化替换与忠实翻译的差异；pilot 未出现系统性实体替换、明显 meaning divergence 或训练/验证口径冲突。
- 以上门槛通过后 TD-16D 才能开始；未通过时继续语料修复，不用增加训练步数掩盖数据问题。

# task TD-16: 执行 M2 human-only/distilled 等预算训练

状态：pending

依赖：TD-05、TD-08、TD-12、TD-13、TD-14、TD-15

## 目标

从同一初始 student 在冻结的等预算契约下完成 human-only 与 Hy-MT2 distilled 两组 M2 训练，只用 dev 选出唯一候选，并仅对该候选执行一次正式 test。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-15 共同 cohort、两份 recipe、等预算校验和 dev 选择规则
- TD-14 唯一冻结的训练资源 profile 与运行时容量校验
- TD-10～TD-13 训练、恢复和评测链

## 原子边界

本 task 只执行已冻结的两组 M2 实验；不修改 teacher 数据、不临时调参、不追加任一组预算，也不为比较两组而对 test 运行两次。

## 执行事项

- 启动前验证 cohort、两份 recipe、corpus/teacher manifest、tokenizer、模型/训练配置、代码、依赖、Git 状态和运行命令。
- 从同一初始 state-dict hash 分别训练 human-only/distilled，使用相同 source 曝光、路由权重和 optimizer-step 预算。
- 按相同频率执行 dev loss/生成评测与原子 checkpoint；组内和组间选择只依据 TD-15 冻结规则。
- 监控 NaN/Inf、OOM、方向曝光、截断、吞吐和显存；中断只能从已验证 checkpoint 恢复并记录边界。
- 分别冻结两组最佳 dev checkpoint，离线重载并验证权重、配置、49,152 ID 空间和固定 dev 生成，再输出逐路由/聚合 A/B。
- 按冻结规则选唯一候选；distilled 未优于 baseline 或触发任一路由红线时选择 human-only、记录负结果并停止扩量。
- 唯一候选冻结后只运行一次正式 test，输出 18 路由、12 产品方向及随机初始化基线对照。
- 验证两组 train/dev loss 全程有限且最终 dev loss 低于同协议随机基线；异常运行不得被另一组成功掩盖。
- 明确记录空/弱方向、繁体差距和已知限制，不把 loss 下降单独描述为可发布翻译质量。

## 产物

- 两组 M2 HF checkpoint、run manifest、训练/恢复日志。
- 等预算 A/B 与 dev 选择报告。
- 唯一最终候选及一次性正式 test 报告。

## 验收

- plan 的 M2 门槛全部满足，两组严格遵守等预算契约。
- 最终候选完全由预先冻结的 dev 规则选出。
- test 未参与训练、调参、checkpoint 或组间选择，且只执行一次。
- 蒸馏负结果会如实保留并阻止 teacher 数据扩量。

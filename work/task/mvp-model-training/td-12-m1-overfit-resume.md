# task TD-12: 完成 M1 小样本过拟合与恢复验收

状态：pending

依赖：TD-05、TD-11

## 目标

使用正式 student 与 20 路微型 fixture 证明模型能学习目标映射、保持语言控制、精确恢复并离线保存/重载，从而关闭 M1。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 冻结的 20 路 fixture/composite manifest
- TD-09～TD-11 的 student、训练和 checkpoint/resume 实现

## 原子边界

本 task 只在微型 fixture 上做过拟合和恢复验收，不把结果解释为真实翻译质量，不调优正式 GPU profile，也不读取正式 test。

## 执行事项

- 用正式 `mvp_e8_d2_v48k` 与固定 fixture 建立随机初始化基线，在看结果前冻结 step/token 预算、解码配置和阈值。
- 在预算内将 mean loss 降至初始基线 10% 以下；每路由至少一条固定记忆样例在固定解码下得到正确目标语言和规范化 exact match。
- 确认 20 路均被采样；路由饿死、错目标语言或空输出视为失败，source-copy 按跨语言/简繁互转各自合同判断。
- 从中途 checkpoint 恢复完成相同预算，与连续运行比较 step、采样、loss、权重和固定生成。
- 保存并离线重载最终 HF checkpoint，验证 tokenizer 未修改、词表仍为 49,152、generation config 完整。
- 记录显存峰值、吞吐、耗时、loss 曲线和固定样例，并标注该结果不代表真实质量。

## 产物

- M1 过拟合 HF checkpoint。
- 连续/恢复对照、20 路生成回归与资源报告。
- M1 验收记录。

## 验收

- plan 的 M1 loss、记忆、路由和恢复门槛全部满足。
- 连续与恢复运行满足 TD-11 的一致性契约。
- 最终 checkpoint 可完全离线重载且 tokenizer 根未变化。
- 未通过前不得启动 TD-14 的正式 GPU profile 冻结。

# task TD-16C: 训练新 M0 human foundation

状态：pending（TD-02～TD-05 v3 未完成，当前不活动）

依赖：TD-05 v3、TD-16A、TD-16B 诊断结论

## 目标

从零初始化 `mvp_e8_d2_v48k`，使用 TD-05 新验收的通用 MT human train/dev 训练 foundation，并只用新 human dev 选择最佳 checkpoint。

## 原子边界

本 task 不负责调研、构建或修复语料；这些工作必须在 TD-02～TD-05 完成。本 task 不使用 teacher target、不访问 test、不做 CT2 转换。

## 执行事项

- 在看到长训结果前冻结 token/墙钟上限、learning-rate schedule、评测频率、early-stop 和逐路由质量红线。
- 记录每路由曝光、独立 semantic group 覆盖、累计 token、有效数据遍数、训练/dev loss、BLEU/chrF、实体/数字忠实度、吞吐、显存和墙钟。
- 使用 TD-16A 的配置化高吞吐路径，以最短 time-to-quality 为目标；硬件数值只存在于独立 YAML/本机 profile。
- 定期发布 complete checkpoint；中断从身份匹配 checkpoint 恢复，不要求恢复后权重 hash 与连续运行相同。
- 按总体与20路 dev 能力、脚本合规、实体/数字忠实度、空输出和 source-copy 冻结唯一 human foundation。

## 产物与验收

- 新 M0 的完整训练运行、最佳 dev checkpoint、学习曲线和20路能力报告。
- loss 有限、无路由饥饿，dev 不呈现系统性 entity substitution，early-stop 在过拟合前生效。
- 该 checkpoint 是 TD-16D 的唯一初始化输入，但尚不称为最终发布 MVP。

# task TD-16D: 训练修复语料上的 human foundation

状态：pending（等待 TD-16C 新 corpus）

依赖：TD-16C

## 目标

从零初始化 `mvp_e8_d2_v48k`，使用 TD-16C 新冻结的 human train/dev 训练真正的通用翻译 foundation，并只用新 human dev 选择最佳 checkpoint。

## 原子边界

本 task 不使用 teacher target，不访问 test，不做 CT2 转换。资源数值来自独立训练 YAML 和运行时探测；训练器实现中不写死本机硬件。

## 执行事项

- 在看到长训结果前冻结 token/墙钟上限、learning-rate schedule、评测频率、early-stop 和逐路由质量红线。
- 记录每路由曝光、独立 semantic group 覆盖、累计 token、有效数据遍数、训练/dev loss、BLEU/chrF、实体/数字忠实度、吞吐、显存和墙钟。
- 使用 TD-16A 的缓存、分桶、异步传输和 fused optimizer 等可配置路径，以最短 time-to-quality 为目标；GPU/TDP 只作为诊断，不是单独优化目标。
- 定期发布 complete checkpoint；中断从身份匹配 checkpoint 恢复，不要求恢复后权重 hash 与连续运行相同。
- 以总体与20路 dev 能力、脚本合规、实体/数字忠实度、空输出和 source-copy 选择唯一 human foundation。

## 产物与验收

- 修复后 human corpus 的完整训练运行、最佳 dev checkpoint、学习曲线和20路能力报告。
- loss 有限、无路由饥饿；dev 不再呈现由语料口径造成的系统性 entity substitution，且 early-stop 在过拟合前生效。
- 该 checkpoint 是 TD-16E 的唯一初始化输入，但尚不称为最终发布 MVP。

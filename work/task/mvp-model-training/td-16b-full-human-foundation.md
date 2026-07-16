# task TD-16B: 训练完整 human M0 底模

状态：pending

依赖：TD-16A

## 目标

从零初始化 `mvp_e8_d2_v48k`，使用完整 226,218 条、20 路 human M0 train 语料训练真正的 human foundation，并只用冻结 human dev 选择最佳 checkpoint。

## 原子边界

本 task 不使用 teacher target，不访问 test，不做 CT2 转换。训练停止由 TD-16A 冻结的 token/墙钟上限、dev 改善和 early-stop 规则共同决定，不再把固定 1,000 步本身视为完成门槛。

## 执行事项

- 验证完整 human M0 manifest、20 路记录数、tokenizer、student 配置和训练资源边界。
- 从零训练并记录每路由曝光、累计 token、有效 epoch、训练/dev loss、BLEU/chrF、吞吐、显存和墙钟。
- 定期发布可加载 checkpoint；中断可从最近 complete checkpoint 继续，但不要求恢复后权重 hash 与连续运行一致。
- 以总体与逐路由 dev 能力、脚本合规和错误率选择 human foundation；弱路由不得被总体均值掩盖。
- 在达到 early-stop 或冻结上限后发布唯一 human foundation HF checkpoint。

## 产物与验收

- 完整 human M0 训练运行、最佳 dev checkpoint、学习曲线和20路能力报告。
- 226,218 条记录均处于可采样范围，20路曝光无饥饿、loss 有限、零截断或截断符合冻结门槛。
- 该 checkpoint 是 TD-16C 的唯一初始化输入，但尚不称为最终发布 MVP。

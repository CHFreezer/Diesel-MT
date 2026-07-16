# task TD-16B: 验证完整 human M0 长训并定位语料边界

状态：completed（诊断完成，候选不准入后续训练）

依赖：TD-16A

## 目标

使用完整 226,218 条、20 路 human M0 train 选择范围执行从零长训，判断它能否直接形成通用翻译 human foundation；只使用 human dev 诊断，不访问正式 test。

## 运行事实

- 训练稳定、loss 有限、20 路均有曝光，优化后的端到端吞吐约为 33.8k tokens/s；训练器不是本轮失败的首要原因。
- 训练内 validation loss 在 step 4,000 达到最低 `3.867906`，随后持续回升；step 10,000 为 `4.114627`，step 15,000 为 `4.209844`。按用户授权在完整 step-15,000 checkpoint 发布后 early-stop。
- 5k/10k/15k 三个 checkpoint 均完成隔离 HF 转换和 dev-only 生成评测。按逐路由优先规则，step 10,000 的 macro route chrF `25.8825` 高于 5k/15k，故只作为本地诊断候选保留；正式 test 未运行。
- 226,218 条 directed records 实际来自 11,411 个 semantic/alignment groups，平均每组扩展为 `19.82` 条路由记录；共有 54,425 个 `(language, text)` 唯一文本。step 10,000 已消费 1,920,000 条样本，约为 directed train 的 `8.49` 遍、每个语义组平均约 `168` 次跨路由曝光。

## 语料结论

- 当前 M0 全部来自 MASSIVE 1.1，定位是多语言虚拟助手本地化，不是通用逐字/忠实翻译语料。仓库验收报告原本也只允许它用于 route/training-system validation，不支持生产翻译质量结论。
- 数据管线读取 `text_field: utt`。MASSIVE 合法的 locale adaptation 会替换地点、媒体、服务和人物；例如英文城市/机构在日文 target 中改成本地城市/机构。对意图分类这是正确本地化，对通用机器翻译却会直接教坏实体忠实度。
- 旧 manual review 已记录截断、meaning divergence、action omission 等已知警告，但当时把 locale-specific entity substitution 视为可接受；该口径不满足当前“通用翻译底模”目标。

## 验收结论

TD-16B 完成的是一次证据化否决：训练方法能够工作，但旧 M0 的独立语义规模、领域宽度和本地化对齐口径不足以承担通用 MT foundation。现有 5k/10k/15k 权重与评测只保留为诊断证据，不进入蒸馏、重复训练、正式 test 或 MVP 发布；关键路径回退到 TD-16C 重新准备语料。

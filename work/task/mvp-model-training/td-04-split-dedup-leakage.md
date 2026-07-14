# task TD-04: 实现分组切分、去重与泄漏防护

状态：pending

依赖：TD-03

## 目标

在正反方向扩展前建立稳定 group 级 split、跨集合去重和污染检查，确保任何对齐关系及其派生样本不会跨 train/dev/test 泄漏。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-03 的规范样本、provenance、fixture 和构建接口
- tokenizer corpus/holdout 与计划使用的正式 MT 评测集身份

## 原子边界

本 task 只实现和验证 split/dedup/leakage 算法；真实规模 M0 双构建、人工抽检和发布决定由 TD-05 完成。

## 执行事项

- 在扩展正反方向前按无向平行关系生成稳定 group；同一对齐关系、反向样本、同文档片段和已知派生样本必须同 split。
- 在规范文本、source、target 和 pair 层做 exact 去重，并对 train/dev/test 做 near-duplicate/污染检查，记录算法、参数和命中原因。
- 使用稳定 group hash 和版本化比例生成 split，禁止逐行随机切分；在构建阶段冻结 test 身份。
- 与 tokenizer corpus/holdout、正式 MT 评测集及同源重复版本做可追溯污染检查，不把 tokenizer holdout 当模型 test。
- split 后再扩展正反方向，验证 18 路由不会因反向关系跨 split 泄漏。
- 验证 worker、缓存、输入完成顺序和 fresh/resume 不改变 corpus、split 或 manifest 字节。
- 增加反向泄漏、近重复、派生样本、错误 group 和非确定顺序失败测试。

## 产物

- 确定性 split/dedup/leakage 模块。
- 污染命中与拒绝报告 schema。
- 泄漏和复现自动化测试。

## 验收

- train/dev/test 在 group 层严格隔离，test 身份稳定冻结。
- 所有反向和派生关系均被绑定在同一 split。
- fresh/resume 与不同 worker 顺序产生字节级相同结果。
- 任一泄漏或污染红线命中都会明确阻止发布。

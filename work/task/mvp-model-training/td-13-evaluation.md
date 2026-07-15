# task TD-13: 实现独立评测与方向汇总

状态：pending

依赖：TD-05、TD-09

## 目标

实现与训练进程解耦的离线评测入口，以固定协议输出 20 路明细、12 个跨语言产品方向汇总和 2 个简繁互转结果，并严格隔离 dev/test 使用边界。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 冻结的 dev/test、路由和 split manifest
- TD-09 tokenizer/model 加载与生成语义
- 随机初始化和后续 M1/M2 HF checkpoint

## 原子边界

本 task 只实现评测协议、指标和报告，不训练或选择模型；test 入口存在但默认禁止训练过程中访问，正式 test 调用由 TD-16 控制。

## 执行事项

- 实现 `scripts/evaluate_mvp_model.py`，离线加载数据、tokenizer 和 HF checkpoint，分离 dev/test 入口并默认拒绝训练期 test 访问。
- 锁定 SacreBLEU/chrF 版本、tokenization/signature、文本规范和生成参数，并记录可复现命令。
- 报告 loss、SacreBLEU、chrF、目标脚本合规、空输出、source-copy、长度比、截断率和固定样例。
- 先输出 20 路明细，再汇总 12 个跨语言产品方向并单列 2 个简繁互转结果；中文汇总保留 `zho_Hans`/`zho_Hant` 明细和样本权重。
- 对随机初始化、M1 和 M2 候选使用同一协议，禁止用 train 样本冒充 dev/test 质量。
- 保存逐样本输出、汇总 JSON、Markdown 报告以及配置/模型/数据哈希关联。
- 增加指标、脚本合规、20 路到 12+2 汇总、空 split、错标签和 test 访问边界测试。

## 产物

- `scripts/evaluate_mvp_model.py` 与指标/汇总模块。
- 固定评测配置、逐样本/JSON/Markdown 报告 schema。
- 独立评测自动化测试。

## 验收

- 任意合法 checkpoint 在相同数据/生成配置下产生可复现结果。
- 20 路、12 个跨语言产品方向与 2 个简繁互转结果均完整，繁简明细不会被聚合掩盖。
- test 访问需要显式授权且训练流程无法调用。
- 报告可追溯到 checkpoint、数据和配置身份。

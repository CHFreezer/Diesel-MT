# task TD-13: 实现独立评测与方向汇总

状态：completed

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

## 完成记录

- 新增 `configs/mvp_evaluation.yaml`、`scripts/mvp_evaluation.py` 和独立 CLI `scripts/evaluate_mvp_model.py`。配置严格冻结 dev/test 数据身份、每路前 10 条规范顺序、128/128 编码、greedy decode、SacreBLEU 2.6.0 与 chrF 参数；CLI 默认 dev，test 必须显式传入 `--allow-test`。
- 报告同时包含 20 个 tag route、12 个产品方向和 2 个简繁转换方向。中文产品汇总保留参与的 `zho_Hans`/`zho_Hant` tag route 及样本权重，逐样本输出保留 loss 权重、脚本合规、空输出、source-copy、target control、长度比和截断状态。
- 使用 TD-12 M1 checkpoint 对冻结 dev 每路 10 条、共 200 条完成真实离线评测。checkpoint state SHA-256 为 `3cfc2ba0d33afb05f5ec26b4a132f9b491548d58ab55ec13910da36ffabc8273`；overall loss `13.34375`、SacreBLEU `0.894812410614293`、chrF `5.60209852603419`，target control 与脚本合规均为 `1.0`，空输出/source-copy/source 截断/target 截断均为 `0.0`。低质量分数符合 M1 仅用于单样例记忆验收的边界，不解释为已训练翻译模型。
- 两次独立 dev 评测的 `samples.jsonl`、`summary.json`、`report.md` 和 `manifest.json` 均逐字节一致；对应 SHA-256 分别为 `2d28c044ab3accae3a31cc14dc611abf114769554e1e7007616a1a3ddaf4e337`、`f2a2ec7f96ec325cdaad767f0a607b3afe9bd838c44557c511ce772b143e7761`、`de15bc2f547bbad6c6fbf91488635c7eaa7de6fc5cbe3e40ca58eba4ecaa5ab3`、`d5288168cb17e7c8aa69e4a9dfa3b27cfa1e8bdaf269bc7c68bfb15cf8b3cba`。
- 机器记录为 `artifacts/model-training/reports/student/evaluation-protocol.json`，运行时发布根为 `D:\Diesel-MT-Runtime\td13-m1-dev-v1`；自动化覆盖指标签名、五标签脚本判定、20→12+2 聚合、空/缺路由、配置漂移、原子发布和 test 授权边界。

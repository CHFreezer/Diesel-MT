# task TD-04: 实现分组切分、去重与泄漏防护

状态：completed

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

## 完成记录

- 实现 [`finalize_model_data.py`](../../../scripts/finalize_model_data.py) 薄 CLI 与 [`model_data_split_pipeline.py`](../../../scripts/model_data_split_pipeline.py) 核心模块，严格验证 TD-03 complete manifest、规范 corpus 大小/SHA-256 和无向输入方向。
- split profile 冻结为与锁定 MASSIVE 规模一致的 16,521 个稳定 SHA-256 component 桶：train/dev/test=`11514/2033/2974`；profile SHA-256 为 `935583630fcd06d39b9cf5c89bac92a76ef4e56d33ccfe1b3c11f030d9ecff0d`。
- 同 `(source,version,alignment_key)`、exact 语言文本、确认的 near-duplicate 和显式派生链接先绑定到同一 component，再分配 split。错误 alignment group、派生 parent/child 不同 group、输入 group 跨 split 均 fail-fast。
- exact 层按 language/text、source、target 和无向 pair 记录碰撞，并只对两侧文本都相同的 pair 保留确定性 winner；near profile 使用 casefold/空白折叠后的字符 trigram Jaccard `0.82`，小集合穷举，大集合用确定性 MinHash LSH 产候选再精确确认。profile SHA-256 为 `5a1798cf8592e27d6da529169cff3f1a2c3394da3bc375b2d5951f94eb09f68d`。
- split 后才扩展 forward/reverse，输出按 split、18 路由、group、sample ID 固定排序；二次审计阻断 group、反向关系、exact 或 near 文本跨集合。`test-groups.jsonl` 和其哈希冻结 test 身份。
- 新增 [`mvp_model_contamination.yaml`](../../../configs/mvp_model_contamination.yaml)，锁定 tokenizer corpus、tokenizer holdout 和 tokenizer evaluation manifest 身份并全部作为 report-only；tokenizer holdout 明确不作为模型 test。大型参考文件按锁定大小/SHA-256 流式扫描。
- `mt_evaluation` 和 `same_source_version` 强制 `policy=block`；block hit 只发布诊断报告而不发布 complete manifest。正式 CLI 默认要求完整 registry，当前 `formal_mt_evaluation=pending_td05` 会在大型扫描前快速阻断；只有显式开发开关可绕过，且不能用于 M0。
- 完整规则、产物 schema 和运行边界见 [`model-data-split-leakage.md`](../../../docs/model-data-split-leakage.md)。fixture 覆盖反向泄漏、exact/near 跨 split、错误 group、派生链接、输入乱序、外部 report/block policy、半成品发布和字节级复现。
- 专项验证：`.conda\python.exe -m pytest tests/test_model_data_split_pipeline.py tests/test_model_data_pipeline.py tests/test_model_training_contract.py -q`，结果 `47 passed in 0.97s`。
- 全量离线验证：`.conda\python.exe -m pytest -q`，结果 `99 passed in 38.02s`。
- 正式 MT evaluation 身份锁定、40,251,390-byte MASSIVE 构建、全量 tokenizer/评测污染扫描、不同 cache/worker 的真实规模双构建、人工抽检和 M0 发布决定仍属于 TD-05；TD-04 完成不表示正式 corpus 已发布。

本 task 未单独创建 review；统一 review 仍由 TD-18 负责。

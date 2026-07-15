# MVP 模型数据切分、去重与泄漏防护

TD-04 的 v1 已消费 9 组规范无向样本并发布 18 个有向路由。2026-07-16 范围修正后本 task 退回 `pending`：新版本必须消费 10 组关系，在反向扩展前完成相同的 group/component 绑定、确定性 split、精确去重、近重复隔离和外部污染检查，再发布 20 个有向路由；原 v1 M0 继续保持不可变历史证据。

## 入口与正式门禁

无副作用检查：

```pwsh
.conda\python.exe scripts/finalize_model_data.py --dry-run
```

正式构建默认要求 `configs/mvp_model_contamination.yaml` 已包含锁定的正式 MT evaluation 身份和所有已知同源重复版本。TD-05 已锁定原版 FLORES-200 `dev/devtest`；不带开发开关的实际构建会校验并扫描该引用。

仅 fixture/算法开发允许显式绕过完整性门禁：

```pwsh
.conda\python.exe scripts/finalize_model_data.py --allow-incomplete-references
```

这个开关产生的结果不能作为 M0 正式 corpus。任何 `mt_evaluation` 或 `same_source_version` reference 都必须使用 `policy=block`；不能通过改成 report-only 绕过红线。

## 稳定 group/component split

- TD-03 的同一 `(source,version,alignment_key)` 必须只有一个 `sample_group_id`，输入 group 已跨 split、错误 alignment group 或派生样本与 parent 不同 group 都立即失败。
- 完全相同的 `(language,text)` 和达到近重复阈值的文本会先把相关 group 合并为稳定 split component，再分配 split。因此共享文本、反向样本和已知派生关系不能依靠独立哈希落入不同集合。
- component ID 是排序 group ID 列表的规范 JSON SHA-256；split 是 `split-profile version + component ID` 的 SHA-256 稳定桶，不读取输入顺序、worker、cache、绝对路径或时间。
- split profile 固定为 MASSIVE 1.1 锁定分区规模的 `11514/2033/2974` 个 train/dev/test 桶，共 16,521 桶；profile SHA-256 为 `935583630fcd06d39b9cf5c89bac92a76ef4e56d33ccfe1b3c11f030d9ecff0d`。
- `test-groups.jsonl` 单独冻结 test group/component 身份，构建报告同时记录其 SHA-256。

未来来源若存在同文档片段或新的派生关系，adapter/registry 必须以 `derived_sample_links` 显式给出 child/parent sample ID；TD-04 不根据相邻行或文件顺序猜测文档关系。MASSIVE 当前以单条对齐 utterance 为原子记录，不存在更细的文档片段层级。

## Exact 与 near-duplicate

Exact 阶段按语言文本绑定 component，按无向标签对和两侧规范文本保留唯一、最小稳定 sample ID；source/target collision 另行计数，避免把“一侧相同但另一侧不同”的可选翻译静默删除。

Near 阶段只用于隔离和污染诊断，不改写输出文本。检测规范为 Unicode casefold + 空白折叠后的字符 trigram Jaccard：

- 阈值 `0.82`，至少 5 个 trigram；
- 每种语言不超过 2,000 个唯一文本时穷举比较；
- 更大集合使用 24-permutation、6-band 的确定性 MinHash LSH 生成候选，再用真实 Jaccard 确认；
- dedup profile SHA-256 为 `5a1798cf8592e27d6da529169cff3f1a2c3394da3bc375b2d5951f94eb09f68d`。

split 后执行第二次审计：任一 group、forward/reverse relation、exact 文本或确认的 near-duplicate 跨 train/dev/test 都阻止发布。`zho_Hans--zho_Hant` 必须与同一 MASSIVE alignment group 的其他关系保持同 split。

## 外部污染 registry

`configs/mvp_model_contamination.yaml` 当前锁定以下运行时 manifest 身份：

- tokenizer MVP corpus：`tokenizer_corpus`，report-only；
- tokenizer MVP holdout：`tokenizer_holdout`，report-only；
- frozen tokenizer evaluation：`tokenizer_evaluation`，report-only。
- 原版 FLORES-200 `dev/devtest`：`mt_evaluation`，block。

前三类集合用于审计 tokenizer 训练/评估暴露，不是模型质量 test。特别是 tokenizer holdout 不会被重命名或复用为 MT test。由于这些文件超过 2.4 GB，它们只做大小/SHA-256 校验与流式 exact 扫描；原版 FLORES-200 使用 exact + near 扫描并阻断命中。候选索引只为模型数据建立。

正式 M0 对 FLORES-200 的 10,045 条引用记录得到 0 个 exact/near 命中。tokenizer 字符覆盖评测有 2 条与 M0 dev exact 重叠，按冻结的 report-only 语义披露但不把它误称为 MT test 泄漏。如果模型 train/dev/test 任一文本与 block reference exact/near 重叠，管线只写 `td04-contamination-blocked.json` 诊断，不发布 complete manifest。

## 规范输出

- `data/model/corpus/mvp/finalized/train.jsonl`
- `data/model/corpus/mvp/finalized/dev.jsonl`
- `data/model/corpus/mvp/finalized/test.jsonl`
- `data/model/corpus/mvp/finalized/test-groups.jsonl`
- `data/model/reports/td04-dedup-leakage.json`
- `data/model/corpus/mvp/finalized/manifest.json`

split 完成后，每个保留的无向样本才扩展 forward/reverse；v1 输出按 18 路由排序并保持不变，新版本必须按 20 路由固定顺序、group 和 sample ID 排序。所有文件采用 UTF-8/LF、规范 JSONL、同目录临时文件和 `os.replace()`；manifest 最后发布，是唯一完成标记。

专项验证：

```pwsh
.conda\python.exe -m pytest tests/test_model_data_split_pipeline.py tests/test_model_data_pipeline.py tests/test_model_training_contract.py -q
```

# M0 模型训练数据验收

状态：TD-05 completed / `m0-model-training-data-20route-v2`

验收日期：2026-07-15

## 结论

M0 schema v2 已发布为 `status=complete`，覆盖 5 个模型标签、10 个无向关系和 20 个有向路由。最终包含 163,754 个无向关系、327,508 条有向样本，train/dev/test 为 226,218/37,508/63,782；teacher synthetic 与工具转换样本均为 0。M0 v1 的 203,942 条 human train 继续作为不可变 18 路历史语料保留，但只有 v2 身份可以供 TD-09 及后续完整路由验收使用。M0 仍只代表有界训练系统 MVP，不代表生产翻译质量。

语料成熟度不得混用：fixture 只用于测试，smoke 只证明真实流程正确，mvp corpus 才允许产生训练/A-B 结论。TD-08 的 D0 v1 只有 2,263 条 accepted teacher targets，属于蒸馏 smoke。20 路 D1 composite 已引用不可变 D1 v1 和两路简繁 addendum，共 44,361 条 accepted train-only teacher targets；只有该 composite 可以进入 TD-15。

schema v2 运行时完成标记为 `data/model/route20-v2/corpus/mvp/m0-manifest.json`，精简可提交证据为 [`td05-m0-20route.json`](../artifacts/model-training/td05-m0-20route.json)。完整质量分布、人工审查队列和复现逐文件明细保存在 Git-ignored versioned build root；v1 的完成标记和证据保持不变。

## 20 路 v2 验收摘要

- 第 10 组 `zho_Hans--zho_Hant` 最终保留 16,311 个无向关系，train/dev/test 为 11,279/1,868/3,164，反向扩展为两条完整路线。
- 固定队列逐条检查 549 条：accepted 400、rejected 149；其中第 10 组按 20 train、10 dev、10 test、20 rejected 的预算完成审查。38 个边界质量警告继续保留，未发现 systemic blocker。
- FLORES-200 五标签 `dev/devtest` 的 10,045 条外部阻断扫描命中 0；所有 20 路 release gate 通过。
- 冷来源构建与已验证缓存下的完全离线 fresh/resume 构建，10 个规范产物逐文件字节一致。
- TD-03/TD-04/M0 manifest SHA-256 分别为 `113a33afa2ca6f73e8e10fbd5a3dab876dd470fbf0e570320edb0961901fe0c7`、`33a40b305012325657fff8e1620f0edf769e15c4aba8d3a4c413faf8c863e6cd`、`5cc369421a705e2eea0076eec06c2bc12de7f278888df2f1ca9add6250ee1d67`。

## 历史 v1 正式规模

最终保留 147,443 个无向关系，反向扩展后 294,886 条有向样本：train 203,942、dev 33,490、test 57,454。每条路线每 epoch 权重为 1.0、最多曝光一次，不对低资源路线做重复采样。

| 无向组 | train | dev | test | 最终无向 | TD-03 rejected | TD-04 去重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eng_Latn--jpn_Jpan` | 11,406 | 1,870 | 3,212 | 16,488 | 19 | 14 |
| `eng_Latn--kor_Hang` | 11,414 | 1,873 | 3,214 | 16,501 | 9 | 11 |
| `eng_Latn--zho_Hans` | 11,370 | 1,866 | 3,207 | 16,443 | 50 | 28 |
| `eng_Latn--zho_Hant` | 11,241 | 1,846 | 3,155 | 16,242 | 256 | 23 |
| `jpn_Jpan--kor_Hang` | 11,298 | 1,861 | 3,203 | 16,362 | 9 | 150 |
| `jpn_Jpan--zho_Hans` | 11,312 | 1,858 | 3,193 | 16,363 | 8 | 150 |
| `jpn_Jpan--zho_Hant` | 11,305 | 1,858 | 3,175 | 16,338 | 32 | 151 |
| `kor_Hang--zho_Hans` | 11,310 | 1,857 | 3,193 | 16,360 | 4 | 157 |
| `kor_Hang--zho_Hant` | 11,315 | 1,856 | 3,175 | 16,346 | 24 | 151 |

所有组的 train/dev/test 都超过冻结最低门槛。简体与繁体分别来自 MASSIVE `zh-CN`/`zh-TW`，各自具有独立 dev/test 身份；报告按两种脚本分别保留来源、长度、长度比、脚本合规和过滤统计。

## 外部污染与评测边界

原版 FLORES-200 只作为外部阻断引用，不进入 M0。锁定的五标签 `dev/devtest` 共 10,045 条，exact + near 命中为 0。tokenizer 字符覆盖评测中有 2 条与 M0 dev exact 重合；它不是 MT 质量 test，因此按 registry 的 report-only 语义披露，不阻断 M0。

评测选择与锁定细节见 [`model-training-dataset-research.md`](model-training-dataset-research.md#td-05-独立评测污染引用结论)，准备入口为 `scripts/prepare_mt_evaluation.py`。

## 人工抽检

固定队列 SHA-256 为 `cbb1a3c47cf963026a92b0562076ec650fad321167d4eb03c8e94851201bb74d`，共逐条检查 489 条：

- accepted 360 条：每组 train 20、dev 10、test 10；
- rejected 129 条：每组最多 20，不足则全检；
- 9 条专门覆盖 `zho_Hant` 混合脚本，抽样还强制包含短、长和长度比边界；
- 所有 rejected 都符合记录的冻结过滤规则；
- accepted 中标记 29 个明显截断、动作缺失、算术变化或跨 locale 含义错位问题。

这 29 条来自刻意偏向边界的样本，不能解释为总体缺陷率。它们再次确认 MASSIVE 是本地化的虚拟助手语料，适合训练系统和语言控制验证，但不能支撑生产质量声明。审查没有删除或手工修改已锁定 corpus；完整 review ID、问题分类和限制写在 [`m0_manual_review.yaml`](../configs/m0_manual_review.yaml)。

## 字节复现

首次构建使用冷缓存/网络获取与 fresh 输出；第二次使用已验证热缓存、禁止网络的新输出根，并再次验证五 locale resume checkpoint 全命中。TD-03/TD-04 是刻意串行的确定性数据路径，没有可变 worker 参数。

两次构建对以下 10 个规范产物逐文件大小/SHA-256 完全一致，包括 83 MB 无向 corpus、train/dev/test、test groups、两个 manifest 和 TD-03/TD-04 确定性报告。关键身份：

- TD-03 manifest：`8af101ed5f5b003fdc56ca439cd16d61579d375e3c9e7f78ef412f038bc7b761`；
- finalized manifest：`6977224aa904a205b8082adaf8913c28f04cfae0d150706e1664412cb439ec8c`；
- M0 manifest：`3d517a8adad0871d04f688f8fd50e0e6432ea0738a043bb8977f1dcede3c37aa`；
- M0 acceptance report：`4b64f8f3562978332396c11f6708a2ba507442d230d1615a5a36a09f04a40b1a`。

## 复现命令

```pwsh
.conda\python.exe scripts/prepare_mt_evaluation.py --offline
.conda\python.exe scripts/prepare_model_data.py --out data/model --cache-dir data/model/cache --offline --use-cache --resume
.conda\python.exe scripts/finalize_model_data.py
.conda\python.exe scripts/accept_m0_dataset.py --prepare-review
.conda\python.exe scripts/accept_m0_dataset.py
```

# 中文简繁能力合同

决策日期：2026-07-16

状态：20 路范围修正已冻结；实现与数据补充进行中

## 产品定义

Diesel-MT 仍然只有中文、英文、日文、韩文 4 种产品语言，但中文提供两个可独立选择的产品状态：

- `zho_Hans`：简体中文；
- `zho_Hant`：繁体中文。

界面、模型标签和 teacher prompt 继续沿用“简体中文 / 繁體中文”以及 `Chinese` / `Traditional Chinese`，本次范围修正不增加 locale-specific 名称、控制 token 或 tokenizer 标签。当前人类数据分别来自 MASSIVE `zh-CN`/`zh-TW`，该 locale 事实只进入 provenance、来源限制和评测说明，不改变模型语言名称。

完整能力矩阵为：

| 口径 | 数量 |
| --- | ---: |
| 产品语言 | 4 |
| 模型语言标签 | 5 |
| 跨语言产品翻译方向 | 12 |
| 跨语言模型路由 | 18 |
| 中文内部转换/本地化路由 | 2 |
| 完整模型路由 | 20 |
| 产品可选操作 | 14 |

数据合同使用“10 组无向模型关系”，而不是把第 10 组称为另一种语言或普通跨语言翻译：

- 9 组跨语言翻译关系，反向扩展为 18 条跨语言路由；
- 1 组 `zho_Hans--zho_Hant` 中文内部转换关系，反向扩展为两条简繁互转路线。

同标签 identity route 仍然非法。涉及产品级中文时可以简称“中文”；涉及数据、训练、推理和指标时必须写明 `zho_Hans` 或 `zho_Hant`。

## 数据与质量边界

锁定的 MASSIVE 1.1 归档已经包含逐 `(partition,id)` 对齐的 `zh-CN` 与 `zh-TW` 人工本地化记录。两侧清洗 checkpoint 各有 16,521 条：train 11,514、dev 2,033、test 2,974，因此第 10 组不需要重新下载来源，但必须进入新的 config/lock 身份、split/dedup/leakage 构建和人工验收。

第 10 组不是机械字符替换。其预期行为包括繁简字形、当前人类 reference 中的词汇差异、标点与书写习惯调整，同时保持语义、数字、实体、占位符和命令意图。`source_copy` 对这两条路线不能作为通用硬拒绝：共享汉字、数字、缩写、专名乃至合法不变短句都可能保持相同；过滤器必须结合人类 reference、可转换证据和人工审查区分“合法不变”与“未执行转换”。

人类、teacher synthetic 和工具转换数据继续使用不同 provenance。自动转换数据不得冒充 MASSIVE 的 `zh-CN/zh-TW` 人工本地化，也不得替代原生 dev/test。

## 已冻结 v1 产物

以下产物保持不可变并继续作为有效历史证据，但都只覆盖 9 组/18 条跨语言路由，不能重新解释为完整 20 路能力：

- M0 v1：203,942 条 human train，运行身份由现有 `m0-manifest.json` 与 `artifacts/model-training/m0-dataset-acceptance.json` 冻结；
- D0 v1：2,263 条 accepted teacher targets，只是 18 路真实数据 smoke；
- D1 v1：从 40,032 个候选接受 39,941 条，是 18 路跨语言 distilled MVP，证据为 `artifacts/model-training/td08-d1-distilled-data.json`。

冻结的 `mvp-tokenizer-v0` 已包含 `zho_Hans` 与 `zho_Hant`，本次扩展不增加标签、不改变词表和 token ID，因此不重做 tokenizer。

## 20 路补充与组合发布

不得覆盖、改名或回写上述 v1 artifact。完整 20 路发布按以下顺序执行：

1. 更新机器合同、配置和 fixture，将 allowlist 扩展为 10 组/20 路；语言名称继续使用 `Chinese` / `Traditional Chinese`。
2. 使用同一锁定 MASSIVE 1.1 归档构建第 10 组，重新执行 component split、去重、污染扫描、反向隔离、人工抽检和双构建复现，发布新的 M0 20 路身份或显式 human addendum + composite manifest。
3. 在冻结 dev 上使用现有 `Chinese` / `Traditional Chinese` prompt 校准两条中文内部路线；不改变既有 18 路 prompt 身份，也不重跑无关路由。
4. 为两条新增路线各选择 2,224 个 train-only 候选，每路由 accepted 不少于 2,000；独立完成过滤、人工审查、精确 replay 和 manifest-last 发布。
5. 发布引用 v1 与所有 addendum 的 20 路 composite manifest。只有该 composite 身份可以进入 TD-09 的全路由验收和 TD-15/TD-16 正式 A/B；v1 M0/D1 单独不再具备完整 MVP 输入资格。

## 任务回退

- TD-01：`in_progress`，补充 10 组/20 路机器合同、简繁互转语义与 fixture；
- TD-02：`in_progress`，把现有 MASSIVE 第 10 组写入 registry/lock/覆盖矩阵；
- TD-03～TD-05：`pending`，构建、切分并验收新的 human addendum/composite；
- TD-06：保持 `completed`，teacher artifact/runtime 选型不受影响；
- TD-07：`pending`，使用既有语言名称新增两路校准；
- TD-08：`pending`，生成两路 distilled addendum 并发布 20 路 composite；
- TD-09 及以后继续 `pending`，在新的完整数据门槛关闭前不得启动正式全路由验收或训练。

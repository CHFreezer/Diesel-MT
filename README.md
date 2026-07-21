# Diesel-MT

从零训练的轻量级多语言机器翻译模型，覆盖**中文、英文、日文、韩文**之间的 12 个有向翻译方向，并支持简体中文与繁体中文双向转换。项目计划先以独立 48k tokenizer 和约 58.8M 参数的 M2M100 风格 Encoder-Decoder 完成 MVP；MVP 路线通过后，再训练独立 64k tokenizer 和约 201.5M 参数的正式基线。正式基线的原始 INT8 权重估算约 192 MiB，目标是通过 CTranslate2 部署到 CPU / mobile SoC 本地离线推理。

**一个模型覆盖中英日韩，体积小、推理快、离线可用。**

## 为什么从零训练

现成模型没有同时满足体积、方向覆盖、许可证和 CPU 推理要求的方案：

| 障碍 | 具体情况 |
| --- | --- |
| **方向凑不齐** | OPUS-MT 是方向模型，`ja→zh`、`zh↔ko`、`ja↔ko` 缺失；拼多个单向模型会重复存储共享语言表示 |
| **体积太大** | NLLB-200 600M 权重 2.46 GB；decoder-only 翻译 LLM 普遍 3B–8B，CPU 推理不现实 |
| **资产边界不匹配** | NLLB-200 权重是 `CC-BY-NC-4.0`；即使研究用途允许，也会让最终模型直接继承第三方权重/tokenizer 和相应限制，不符合本项目从零训练、独立资产链的目标 |
| **小模型陷阱** | Hy-MT2 1.8B 1.25Bit 只有 462 MB，但依赖 STQ 自定义量化算子，x86 CPU 退回标量回退后性能不可接受；若改用其他运行时的常规 4-bit 权重量化，体积仍约 1.0–1.13 GB，且不属于本项目的 CTranslate2 CPU 路线 |
| **微型 LLM 不适合翻译** | Decoder-only 做翻译是逐 token 自回归生成，KV cache + 串行解码延迟对 CPU 不友好；同参数规模下，Encoder-Decoder 的双向编码 + 交叉注意力是更强的翻译归纳偏置，微型 LLM 翻译质量通常不如等规模的 Encoder-Decoder |

因此 Diesel-MT 选择从零训练一个覆盖 4 种产品语言、12 个跨语言方向和 2 个简繁互转操作的小型 Encoder-Decoder 翻译模型。目标：

- **一个模型覆盖完整 20 路能力**，通过 5 个语言 token 控制 18 个跨语言路由和 2 个简繁互转路由，不拼接多个单向模型
- **从零训练、来源可审计**：不继承第三方模型权重/tokenizer；每个训练来源分别记录许可、署名、非商业/相同方式共享和再分发边界
- **CPU / mobile SoC 可部署**：正式基线计划使用 CTranslate2 INT8，原始权重估算约 192 MiB；实际模型包体以完成转换后的实测结果为准
- **完整闭环**：数据 → tokenizer → 训练 → 评估 → CTranslate2 部署 → 推理，数据与产物可追溯、checkpoint 可恢复、模型能力可复验

---

## 架构

### 模型

主体采用 M2M100 语义的多语言 Encoder-Decoder Transformer，不复用 `facebook/m2m100_418M` 的 checkpoint、词表或默认尺寸。

- 训练接口对齐 `M2M100ForConditionalGeneration`：Encoder-Decoder + LM head，支持 `labels` 训练和 `generate()` 推理
- 配置使用 M2M100 字段语义：`d_model`、`encoder_layers`、`decoder_layers`、`encoder_ffn_dim`、`decoder_ffn_dim`、`encoder_attention_heads`、`decoder_attention_heads`、`vocab_size`
- 翻译控制：encoder 编码源文本，decoder 通过目标语言 token 和 `forced_bos_token_id` 控制输出语言
- CTranslate2 官方支持 M2M100/NLLB 系列；本项目的 CPU 路线只使用 float32/INT8 和批处理优化
- CT2 的 4-bit 支持是读取 AWQ 预量化权重的专用路径，不是通用 `int4` 转换格式或 CPU compute type；AWQ 4-bit 不能在 CPU 上运行，因此不作为本项目的体积或部署目标

**为什么选 M2M100 语义：**

- Encoder-Decoder 天然适合翻译（源文本编码 ↔ 目标文本生成边界清晰）
- many-to-many MT 语义一个模型覆盖多方向，通过语言 token 显式控制
- 共享中英日韩表示，避免多个单向模型重复存储
- 相比通用 text-to-text（T5），M2M100 的训练和推理接口围绕翻译设计，与 CTranslate2 路径更直接

**其他 Encoder-Decoder 的取舍：** Marian/OPUS-MT（方向模型凑不齐）、mBART（需去噪预训练增加复杂度）、NLLB（不作为本项目的权重/tokenizer 起点）、BART/Pegasus/Whisper（面向其他任务）。

**为什么不选 T5 / T5Gemma / T5Gemma2：** 通用 text-to-text 语义不如 M2M100/NLLB 直接匹配 many-to-many MT 方向控制；T5Gemma 系列架构复杂度和许可证边界更重。

### Tokenizer

从零训练，使用 Transformers 5.x 统一后的 `NllbTokenizer`（BPE + Metaspace），保存为 `tokenizer.json`。

- **MVP tokenizer**：已冻结的 `mvp-tokenizer-v0` 为 49,152 词表，只服务约 60M MVP 训练、评测和部署链
- **正式基线 tokenizer**：计划在 MVP 路线通过、约 200M 正式训练语料范围确定后另行训练 65,536 词表；当前尚无该 tokenizer 的配置、产物或训练结果
- **阶段隔离**：64k 是独立新词表，不在 48k 上原地扩词；正式基线模型随 64k tokenizer 从零初始化，不能复用 MVP embedding 或 checkpoint
- **语言 token**：只加入项目需要的 5 个模型标签：`eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang`
- **输入格式**：`<src_lang> source_text </s>` → `<tgt_lang> target_text </s>`
- **生成控制**：`forced_bos_token_id = target_lang_id`
- **许可证边界**：Transformers tokenizer 软件实现是 Apache-2.0，不使用 Meta NLLB-200 的 tokenizer 文件；语料许可与 tokenizer 软件/文件身份分开审计

### 数据质量与可选蒸馏

首个约 60M MVP 使用经过许可、时间、去重和语义审计的 human-first corpus。DeepSeek 在该流程中只做低成本长上下文辅助审计，不负责自动重译或改写训练数据。

- DeepSeek 输入大量带稳定 ID 的 source-target 句对，只稀疏返回疑似问题 ID、严重度、类别和短理由；没有返回不等于逐条质量认证
- 全量审计前以 canary 和人工样本选择安全上下文长度；全部 flag 人工复核，未标记样本继续分层抽检
- 蒸馏生成语料仍可在 human-first 基线之后补充真实数据稀缺的弱关系、近期实体/术语或 dev 已证实的特定错误，但必须另行授权并与等预算 human-only continuation 做 A/B；首个候选曝光仅为全局约 5%～10%、单弱路由不超过约 20%
- 任何 teacher 都必须先通过路线级质量、费用和许可校准；student 始终从零初始化，不继承 teacher 权重、tokenizer 或架构
- dev/test 始终只使用独立 human reference，最终部署只依赖 Diesel-MT + CTranslate2

---

## 语言与方向口径

项目必须区分产品语言、模型语言标签和实际训练路由：

| 口径 | 数量 | 定义 |
| --- | ---: | --- |
| 产品语言 | 4 | 中文、英文、日文、韩文 |
| 模型语言标签 | 5 | `zho_Hans`、`zho_Hant`、`eng_Latn`、`jpn_Jpan`、`kor_Hang` |
| 跨语言产品翻译方向 | 12 | 四种产品语言两两互译 |
| 无向模型关系 | 10 | 9 组跨语言关系，加 `zho_Hans--zho_Hant` 1 组 |
| 有向模型训练路由 | 20 | 18 条跨语言翻译路线，加 2 条简繁互转路线 |
| 产品可选操作 | 14 | 12 个跨语言方向，加 2 个简繁互转操作 |

`zho_Hans` 和 `zho_Hant` 是中文的两个模型标签，不算两种产品语言。`zho_Hans -> zho_Hant` 与 `zho_Hant -> zho_Hans` 属于中文内部转换，不计入 12 个跨语言翻译方向，但进入训练、评测和部署验收。因此准确口径是“4 种产品语言、5 个模型标签、12 个跨语言方向、2 个简繁互转操作、20 个模型路由”。teacher 名称继续沿用 `Chinese` / `Traditional Chinese`，不增加 locale-specific prompt。

后续文档中的“中文”只用于同时适用于简体和繁体的产品层汇总；数据、配置、训练、推理和评测必须明确写 `zho_Hans`/简体中文或 `zho_Hant`/繁体中文。中文汇总指标必须保留简体、繁体明细，不能用合并均值掩盖某一脚本缺失或退化。

---

## 正式基线计划（未实施）

MVP 达到预注册翻译及格线后，计划使用新训练的 64k tokenizer 从零训练正式基线。以下仅为 `M2M100Config` 目标配置和参数/权重估算，不代表 tokenizer、模型权重或训练结果已经存在。

| 配置项 | 目标值 |
| --- | ---: |
| `vocab_size` | 65,536（计划新训练） |
| `d_model` | 768 |
| `encoder_ffn_dim` | 3072 |
| `decoder_ffn_dim` | 3072 |
| `encoder_layers` | 16 |
| `decoder_layers` | 4 |
| `encoder_attention_heads` | 12 |
| `decoder_attention_heads` | 12 |
| `tie_word_embeddings` | true |

| 部署精度 | 权重大小 | 定位 |
| --- | ---: | --- |
| BF16/FP16 | ~384 MiB | GPU / 存储估算，不是 CPU 主线 |
| INT8 | ~192 MiB | 原始权重估算；CT2 包体待实测 |

按 4 bit/参数估算的 ~96 MiB 只是理论权重存储下限，并不对应可在 CPU 上运行的 CT2 产物。当前锁定的 CTranslate2 4.8.1 通用转换器不提供 INT4 输出，CPU compute types 也不包含 4-bit；CT2 的 4-bit 支持仅面向 AWQ 预量化权重，不能作为 CPU 部署选项。

---

## MVP

MVP 不等于流程冒烟。流程、checkpoint 和部署接口由 fixture/smoke 负责；MVP 必须训练出一个在预注册 human dev 上达到总体与逐路由翻译及格线的约 60M 模型。MVP 配置为 `mvp_e8_d2_v48k`：`d_model = 512`、`ffn_dim = 2048`、8 层 encoder、2 层 decoder 和只在 MVP 链内冻结的 49,152 词表。

### 训练路线

```text
近期且许可明确的 human parallel 来源
→ 5万～10万候选 pilot/实收率与预算
→ 确定性硬过滤、去重和数据集隔离
→ DeepSeek 长上下文整批找错（只返回疑似问题 ID）
→ 90万～130万独立 human pairs
→ 从零训练约 60M human-first MVP
→ 仅在能力评测证明有弱路由时执行一次有界补强
→ 重复能力等价、一次正式 test 和 CTranslate2 INT8 发布
```

| 配置 | vocab | enc layers | dec layers | 参数量 | INT8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mvp_e8_d2_v48k` | 49,152 | 8 | 2 | 约 58.8M | 原始 INT8 权重约 56 MiB |

首轮数据预案是 90万～130万个独立 human parallel pairs，正反展开约 180万～260万条 directed records；扩展上限为 150万～200万 pairs，是否使用由 pilot 实收率和首轮能力结果决定，不能以低质 synthetic 填数。领域覆盖自然对话、新闻/百科/说明文、人工锚点、社区短句、持续本地化、少量现代技术文档和简繁/地区正式繁体；繁体数量允许偏低，但不降低质量门。

DeepSeek 审计先比较约 64K/128K/256K input-token 批次，以严重错误 canary 召回率 ≥95%、flag 人工有效命中率 ≥70%、未标记严重错误率 ≤1% 作为暂定放行门。首轮累计费用预计约 150～350 元，扩展累计约 300～600 元；执行前按[官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)复核，预计超过 600 元时重新报告确认。

验收标准：数据预处理可重复 → tokenizer 稳定编码 4 种产品语言的 5 个标签 → 10 组模型关系形成 20 个训练路由 → 从零训练和 checkpoint 恢复稳定 → human dev 上总体与逐路由 BLEU/chrF、loss、脚本/语言控制、空输出、source-copy、实体/数字/术语门通过 → 重复训练在能力与 time-to-quality 上等价 → 唯一候选才可消费一次正式 test。

---

## 模型调研

| 模型 | 时间 | 许可证 | 关键信息 | 结论 |
| --- | --- | --- | --- | --- |
| [OPUS-MT](https://huggingface.co/Helsinki-NLP) | 2022 | Apache-2.0 / CC-BY-4.0 | CTranslate2 成熟；多为单向模型 | `ja→zh`、`zh↔ko`、`ja↔ko` 缺失 |
| [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) | 2020/2022 | MIT | many-to-many，1.94 GB | 架构参考，权重偏大偏老 |
| [NLLB-200 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | 2022-07 | CC-BY-NC-4.0 | M2M100 架构，2.46 GB | 设计参考，不继承权重/tokenizer |
| [SMaLL-100](https://huggingface.co/alirezamsh/small100) | 2022-11 | MIT | 100 语言，~333M 参数 | 路线可参考，需自定义 tokenizer 代码 |
| [mBART-50 MMT](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) | 2020/2022 | 未标注 | 0.6B，需去噪预训练 | 不微调现成权重 |
| [T5 base](https://huggingface.co/google-t5/t5-base) | 2019/2020 | Apache-2.0 | 通用 text-to-text | 方向控制不如 M2M100 |
| [T5Gemma / T5Gemma2](https://huggingface.co/docs/transformers/en/model_doc/t5gemma) | 2025 | Gemma Terms | 新一代 Encoder-Decoder | 架构和许可偏重 |
| [MADLAD-400 3B](https://huggingface.co/google/madlad400-3b-mt) | 2023 | Apache-2.0 | T5 架构 MT | 证明 T5 可做 MT |
| [SeamlessM4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) | 2023 | CC-BY-NC-4.0 | 语音+文本，~2B | 体积和许可不匹配 |
| [LMT-60 0.6B](https://huggingface.co/NiuTrans/LMT-60-0.6B) | 2025-11 | Apache-2.0 | decoder-only，0.8B | 非 Encoder-Decoder |
| [Hy-MT2 7B GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) | 2026-05 | Apache-2.0 | 翻译专用，Q8_0 约 7.98 GB | 仅作可选补强候选；使用前重新校准 |
| [Hy-MT2 1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) | 2026-05 | Apache-2.0 | 1.25Bit 462MB / Q4 1.13GB | x86 性能不可接受 |
| ALMA-7B / TowerInstruct / Aya-23 / Seed-X | 2023–2025 | 混合 | 7B–8B | 太大，可参考数据 |

OPUS-MT 方向缺口：`ja→zh`、`zh↔ko`、`ja↔ko` 未找到明确公开模型。

---

## 开发环境

Python 3.11 + 仓库根目录 `.conda` 环境。详见 [Python 环境约定](docs/python-environment.md)。

```pwsh
& 'C:\Users\chfre\miniconda3\shell\condabin\conda-hook.ps1'
conda activate (Join-Path $PWD '.conda')
python -m pip install -r requirements.txt
```

## 工作流

项目采用 `plan → todo → task → review → done` 工作流。README 只说明流程语义，不维护具体任务清单；具体内容和验证记录放在对应的 `work/` 文档或代码提交中。

### plan

`work/plan/` 说明阶段要解决的问题、边界、约束和验收标准，用于统一方向，不承载具体执行列表。阶段完成后 plan 保留原位作为决策记录，状态改为 `done / archived`，并链接到 `work/done/` 中的归档材料。

### todo

`work/todo/` 将 plan 拆成可执行的待办项，并明确依赖关系和完成条件。

### task

`work/task/` 将 todo 拆成可独立完成、验证和提交的工作单元。每个 task 应记录输入、输出、验证方式和产物位置。

### review

只有一个 todo 下的全部 task 都完成实现和各自验收后，整个 todo 才进入 review。`work/review/` 保存该 todo 及完整 task 集合的统一复核记录，重点检查可复现性、正确性、数据边界、评估结果和部署风险；不为单个 task 提前创建 review。review 通过后，todo、task 和 review 记录分别归档到 `work/done/` 下的对应目录。

### done

`work/done/` 按 `todo/`、`task/`、`review/` 保存通过统一 review 的完整 todo 工作流；验收叙述合并进对应 task 和统一 review，不额外拆分 report Markdown。需要程序消费的合并 JSON 可与对应生成产物放在 `artifacts/`。done 必须能追溯到代码、实验结果和通过的评审结论，不能只依赖状态字段或口头结论；移入 done 后应同步修正 plan 和归档文档中的相对链接。

## 常用命令

```bash
.conda\python.exe -m pytest -q                          # 全部测试
.conda\python.exe -m pytest tests/ -k test_name         # 单个测试
python scripts/calculate_model_parameters.py            # 估算参数量
python scripts/fetch_tokenizer_datasets.py --profile mvp [--resume]  # 下载语料
.conda\python.exe scripts/validate_ctranslate2_deployment.py --phase all --overwrite  # CT2 部署验收
```

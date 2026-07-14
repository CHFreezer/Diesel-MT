# Diesel-MT

从零训练的轻量级多语言机器翻译模型，覆盖**中文、英文、日文、韩文**之间的 12 个有向翻译方向。M2M100 风格 Encoder-Decoder Transformer，基线目标 ~200M 参数、INT8 量化后约 192 MiB，通过 CTranslate2 部署到 CPU / mobile SoC 本地离线推理。

**一个文件小、跑得快、离线可用的中英日韩翻译模型。**

## 为什么从零训练

现成模型没有同时满足体积、方向覆盖、许可证和 CPU 推理要求的方案：

| 障碍 | 具体情况 |
| --- | --- |
| **方向凑不齐** | OPUS-MT 是方向模型，`ja→zh`、`zh↔ko`、`ja↔ko` 缺失；拼多个单向模型会重复存储共享语言表示 |
| **体积太大** | NLLB-200 600M 权重 2.46 GB；decoder-only 翻译 LLM 普遍 3B–8B，CPU 推理不现实 |
| **许可证卡脖子** | NLLB-200 是 `CC-BY-NC-4.0`，权重和 tokenizer 都不能直接用 |
| **小模型陷阱** | Hy-MT2 1.8B 1.25Bit 只有 462 MB，但依赖 STQ 自定义量化算子，x86 CPU 退回标量回退后性能不可接受；若改用其他运行时的常规 4-bit 权重量化，体积仍约 1.0–1.13 GB，且不属于本项目的 CTranslate2 CPU 路线 |
| **微型 LLM 不适合翻译** | Decoder-only 做翻译是逐 token 自回归生成，KV cache + 串行解码延迟对 CPU 不友好；同参数规模下，Encoder-Decoder 的双向编码 + 交叉注意力是更强的翻译归纳偏置，微型 LLM 翻译质量通常不如等规模的 Encoder-Decoder |

因此 Diesel-MT 选择从零训练一个 4 语言 × 12 方向的小型 Encoder-Decoder 翻译模型。目标：

- **一个模型覆盖 12 方向**，通过语言 token 控制，不拼接多个单向模型
- **从零训练、资产可控**：自训练 tokenizer + 自训练权重，无 CC-BY-NC-4.0 污染
- **CPU / mobile SoC 可部署**：CTranslate2 INT8 量化后模型文件约 192 MiB，支持端侧离线推理
- **完整闭环**：数据 → tokenizer → 训练 → 评估 → CTranslate2 部署 → 推理，每一步可复现

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

**其他 Encoder-Decoder 的取舍：** Marian/OPUS-MT（方向模型凑不齐）、mBART（需去噪预训练增加复杂度）、NLLB（CC-BY-NC-4.0 不可用）、BART/Pegasus/Whisper（面向其他任务）。

**为什么不选 T5 / T5Gemma / T5Gemma2：** 通用 text-to-text 语义不如 M2M100/NLLB 直接匹配 many-to-many MT 方向控制；T5Gemma 系列架构复杂度和许可证边界更重。

### Tokenizer

从零训练，使用 Transformers 5.x 统一后的 `NllbTokenizer`（BPE + Metaspace），保存为 `tokenizer.json`。

- **语言 token**：只加入项目需要的 `eng_Latn`、`zho_Hans`/`zho_Hant`、`jpn_Jpan`、`kor_Hang`
- **输入格式**：`<src_lang> source_text </s>` → `<tgt_lang> target_text </s>`
- **生成控制**：`forced_bos_token_id = target_lang_id`
- **许可证边界**：Transformers tokenizer 软件实现是 Apache-2.0，不使用 Meta NLLB-200 的 tokenizer 文件，无 CC-BY-NC-4.0 污染

### 蒸馏

Hy-MT2 7B（Apache-2.0）作为离线 teacher，Diesel-MT 从零训练的 Encoder-Decoder 作为 student。

- teacher 用于生成中英日韩 12 方向平行样本，优先补齐 `zh↔ja`、`zh↔ko`、`ja↔ko` 等弱方向
- student 从零初始化，不继承 teacher 权重、tokenizer 或架构
- 最终部署只依赖 Diesel-MT + CTranslate2，不依赖 teacher

---

## 目标基线

基线按 `M2M100Config` 字段语义描述，~200M 参数。

| 配置项 | 目标值 |
| --- | ---: |
| `vocab_size` | 64k |
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
| INT8 | ~192 MiB | CTranslate2 CPU 部署目标 |

按 4 bit/参数估算的 ~96 MiB 只是理论权重存储下限，并不对应可在 CPU 上运行的 CT2 产物。当前锁定的 CTranslate2 4.8.1 通用转换器不提供 INT4 输出，CPU compute types 也不包含 4-bit；CT2 的 4-bit 支持仅面向 AWQ 预量化权重，不能作为 CPU 部署选项。

---

## MVP

MVP 阶段只验证路线，不追求最终效果。快速验证配置固定 `d_model = 512`、`ffn_dim = 2048`。

| 配置 | vocab | enc layers | dec layers | 参数量 | INT8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `e12-d3-v48k` | 48k | 12 | 3 | 75.50M | 72 MiB |
| `e12-d3-v32k` | 32k | 12 | 3 | 67.11M | 64 MiB |
| `e8-d2-v48k` | 48k | 8 | 2 | 58.72M | 56 MiB |
| `e8-d2-v32k` | 32k | 8 | 2 | 50.33M | 48 MiB |

验收标准：数据预处理可重复 → tokenizer 稳定编码四语言 → 端到端训练 + checkpoint 可恢复 → 固定测试样例回归检查。

---

## 模型调研

调研时间：2026-07-12（NLLB、CTranslate2、T5 补充：2026-07-13）。

| 模型 | 时间 | 许可证 | 关键信息 | 结论 |
| --- | --- | --- | --- | --- |
| [OPUS-MT](https://huggingface.co/Helsinki-NLP) | 2022 | Apache-2.0 / CC-BY-4.0 | CTranslate2 成熟；多为单向模型 | `ja→zh`、`zh↔ko`、`ja↔ko` 缺失 |
| [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) | 2020/2022 | MIT | many-to-many，1.94 GB | 架构参考，权重偏大偏老 |
| [NLLB-200 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | 2022-07 | CC-BY-NC-4.0 | M2M100 架构，2.46 GB | 设计参考，资产不可用 |
| [SMaLL-100](https://huggingface.co/alirezamsh/small100) | 2022-11 | MIT | 100 语言，~333M 参数 | 路线可参考，需自定义 tokenizer 代码 |
| [mBART-50 MMT](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) | 2020/2022 | 未标注 | 0.6B，需去噪预训练 | 不微调现成权重 |
| [T5 base](https://huggingface.co/google-t5/t5-base) | 2019/2020 | Apache-2.0 | 通用 text-to-text | 方向控制不如 M2M100 |
| [T5Gemma / T5Gemma2](https://huggingface.co/docs/transformers/en/model_doc/t5gemma) | 2025 | Gemma Terms | 新一代 Encoder-Decoder | 架构和许可偏重 |
| [MADLAD-400 3B](https://huggingface.co/google/madlad400-3b-mt) | 2023 | Apache-2.0 | T5 架构 MT | 证明 T5 可做 MT |
| [SeamlessM4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) | 2023 | CC-BY-NC-4.0 | 语音+文本，~2B | 体积和许可不匹配 |
| [LMT-60 0.6B](https://huggingface.co/NiuTrans/LMT-60-0.6B) | 2025-11 | Apache-2.0 | decoder-only，0.8B | 非 Encoder-Decoder |
| [Hy-MT2 7B](https://huggingface.co/tencent/Hy-MT2-7B) | 2026-05 | Apache-2.0 | 翻译专用，~8B | **离线蒸馏 teacher** |
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

`work/review/` 保存正在复核的 task，重点检查可复现性、正确性、数据边界、评估结果和部署风险。实现完成不等于 done；review 通过后，todo、task 和 review 记录分别归档到 `work/done/` 下的对应目录。

### done

`work/done/` 按 `todo/`、`task/`、`review/` 保存已完成实现、验证和评审记录。done 必须能追溯到代码、实验结果和通过的评审结论，不能只依赖状态字段或口头结论；移入 done 后应同步修正 plan 和归档文档中的相对链接。

## 常用命令

```bash
.conda\python.exe -m pytest -q                          # 全部测试
.conda\python.exe -m pytest tests/ -k test_name         # 单个测试
python scripts/calculate_model_parameters.py            # 估算参数量
python scripts/fetch_tokenizer_datasets.py --profile mvp [--resume]  # 下载语料
```

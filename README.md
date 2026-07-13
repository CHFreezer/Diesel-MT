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
| **小模型陷阱** | Hy-MT2 1.8B 1.25Bit 只有 462 MB，但依赖 STQ 自定义量化算子，x86 CPU 退回标量回退后性能不可接受；退回常规 4bit 后约 1.0–1.13 GB，体积优势消失 |
| **微型 LLM 不适合翻译** | Decoder-only 做翻译是逐 token 自回归生成，KV cache + 串行解码延迟对 CPU 不友好；同参数规模下，Encoder-Decoder 的双向编码 + 交叉注意力是更强的翻译归纳偏置，微型 LLM 翻译质量通常不如等规模的 Encoder-Decoder |

因此 Diesel-MT 选择从零训练一个 4 语言 × 12 方向的小型 Encoder-Decoder 翻译模型。目标：

- **一个模型覆盖 12 方向**，通过语言 token 控制，不拼接多个单向模型
- **从零训练、资产可控**：自训练 tokenizer + 自训练权重，无 CC-BY-NC-4.0 污染
- **CPU / mobile SoC 可部署**：CTranslate2 INT8/INT4 量化后 100–200 MiB，端侧离线推理
- **完整闭环**：数据 → tokenizer → 训练 → 评估 → CTranslate2 部署 → 推理，每一步可复现

---

## 架构

### 模型

主体采用 M2M100 语义的多语言 Encoder-Decoder Transformer，不复用 `facebook/m2m100_418M` 的 checkpoint、词表或默认尺寸。

- 训练接口对齐 `M2M100ForConditionalGeneration`：Encoder-Decoder + LM head，支持 `labels` 训练和 `generate()` 推理
- 配置使用 M2M100 字段语义：`d_model`、`encoder_layers`、`decoder_layers`、`encoder_ffn_dim`、`decoder_ffn_dim`、`encoder_attention_heads`、`decoder_attention_heads`、`vocab_size`
- 翻译控制：encoder 编码源文本，decoder 通过目标语言 token 和 `forced_bos_token_id` 控制输出语言
- CTranslate2 官方支持 M2M100/NLLB 系列，提供 CPU/GPU 量化和批处理优化路径

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

| 精度 | 权重大小 |
| --- | ---: |
| BF16/FP16 | ~384 MiB |
| INT8 | ~192 MiB |
| INT4 | ~96 MiB |

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

## 常用命令

```bash
.conda\python.exe -m pytest -q                          # 全部测试
.conda\python.exe -m pytest tests/ -k test_name         # 单个测试
python scripts/calculate_model_parameters.py            # 估算参数量
python scripts/fetch_tokenizer_datasets.py --profile mvp [--resume]  # 下载语料
```

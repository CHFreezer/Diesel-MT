# Diesel-MT

Diesel-MT 是一个面向中文、英文、日文、韩文的轻量多语言翻译模型实验项目。目标是先用小模型跑通 MVP，再扩展到约 200M 参数的基线模型，优先考虑 CPU 和 mobile SoC 本地推理。

## 目标

- 覆盖 **中英日韩** 四种语言之间的 12 个有向翻译方向。
- 使用一个多语言互译模型，而不是拼接多个单向模型。
- 采用 M2M100 风格 Encoder-Decoder 架构。
- 从零训练中英日韩专用 tokenizer，并采用 NLLB 风格语言 token 控制。
- 建立从数据、tokenizer、模型配置、训练、评估到推理的完整 MVP 闭环。

## 背景结论

调研时间：2026-07-12。NLLB-200 架构与 tokenizer、CTranslate2、Encoder-Decoder 和 T5 系列补充核查时间：2026-07-13。

现成模型没有同时满足体积、方向覆盖、许可证、发布时间和 CPU/mobile 推理要求：

- OPUS-MT 方向模型凑不齐中英日韩 12 个方向；即使补齐多个单向模型，也会重复存储英语、中文、日文等共享表示。
- M2M100 / NLLB 的架构有参考价值，但公开 checkpoint 较大；NLLB-200 还带 `CC-BY-NC-4.0` 非商用限制。
- Hy-MT2 1.8B 1.25Bit 原本最像现成首选，但 1.25bit 小体积绑定 STQ 自定义量化算子，退回成熟 4bit 又会失去模型大小优势。
- 近年的 decoder-only 翻译 LLM 通常从 3B 起步，更常见 7B/8B。对 CPU/mobile SoC 来说，权重、KV cache、逐 token 解码延迟都不适合作为本项目基线。

为什么不选择 Hy-MT2：它翻译专用、发布时间新、Apache-2.0、文件只有约 462 MB，表面上最接近本项目想找的现成小模型。但问题在推理路径：

- 462MB 的 Hy-MT2 是“特定 ARM kernel 下的小模型”，不是“通用 CPU 上小而快的模型”。
- 小体积依赖 1.25bit STQ 自定义量化算子；该路线目前的有效性能路径主要面向特定 ARM 平台，更接近 Apple Silicon / MacBook 类设备。
- 在 x86 CPU 上缺少对应 SIMD kernel 时，会退回 scalar fallback。这个路径不具备常规 llama.cpp 量化模型的 CPU 推理性能，不能作为本项目的通用 CPU/mobile 基线。
- 要找回成熟 CPU 推理性能，只能退回 `Q4_K_M` 等常规 4bit 量化；此时 Hy-MT2 约 1.0-1.13 GB。按同样 4bit 部署口径比较，M2M100 418M 和 NLLB-200 distilled 600M 反而更小，Hy-MT2 的主要优势消失。

但 Hy-MT2 仍适合作为蒸馏 teacher：Hy-MT2 7B 是 Apache-2.0 的翻译专用模型，覆盖本项目需要的中英日韩语言。它不进入最终 CPU/mobile 推理链路，只用于离线生成翻译样本、补齐弱方向数据、提供候选译文或做数据过滤。

为什么不选择 SMaLL-100：它证明“小型多语言 M2M100 路线”可行，但不适合作为本项目直接基座。

- SMaLL-100 是作者发布的论文模型，维护和工程生态不如 Meta / Google / Tencent 官方模型稳定。
- 模型卡宣称整体指标有竞争力，但本项目关心的是中英日韩 12 方向；这些方向需要重新独立复测，不能直接把 100 语言平均表现当成本项目结论。
- SMaLL-100 修改了 M2M100 tokenizer 行为，模型卡要求当前从 `tokenization_small100.py` 本地加载 tokenizer；部署时依赖远端自定义代码存在供应链隐患，不能把模型仓库里的 Python 代码当作长期可信执行面。
- HF 文件显示参数量约 333M，已经高于本项目约 200M 的长期基线目标。

因此，Diesel-MT 选择缩小语言范围，从零训练一个 4 语言、12 方向的小型 Encoder-Decoder 翻译模型。

## 架构选择

### 模型

主体采用 M2M100 语义的多语言 Encoder-Decoder Transformer，但不复用 `facebook/m2m100_418M` 的 checkpoint、词表或默认尺寸。

- 训练接口对齐 `M2M100ForConditionalGeneration`：Encoder-Decoder 主体后接语言建模 head，支持 `labels` 训练和 `generate()` 推理。
- `M2M100Model` 只作为架构主体参考；它输出 hidden states，不负责生成翻译 logits。
- 配置直接使用 M2M100 字段语义：`d_model`、`encoder_layers`、`decoder_layers`、`encoder_ffn_dim`、`decoder_ffn_dim`、`encoder_attention_heads`、`decoder_attention_heads`、`vocab_size`。
- 翻译控制沿用 M2M100/NLLB 系列的做法：encoder 编码源文本，decoder 通过目标语言 token 和 `forced_bos_token_id` 控制输出语言。
- 部署目标需要兼容 CTranslate2；CTranslate2 官方支持 M2M100/NLLB 这类多语言 Encoder-Decoder 翻译模型，并提供 CPU/GPU、量化和批处理优化路径。

为什么选 M2M100 语义：

- 翻译任务天然是 source sequence 到 target sequence 的条件生成，Encoder-Decoder 比 decoder-only 更适合固定翻译任务，尤其是源文本编码和目标文本生成边界清晰。
- M2M100 是 many-to-many MT 语义：一个模型覆盖多个源语言和目标语言，通过语言 token 显式控制方向，正好对应中英日韩 12 个有向方向。
- 相比 OPUS-MT/Marian 这类方向模型，M2M100 语义能共享英语、中文、日文、韩文的表示，避免多个单向模型重复存储相近语言知识。
- 相比通用 text-to-text 架构，M2M100 的训练和推理接口已经围绕多语言翻译设计，和 CTranslate2 的翻译部署路径更直接。

为什么不选 Google T5 / T5Gemma / T5Gemma2 作为主线：

- T5 是通用 text-to-text Encoder-Decoder，不是专门为 many-to-many MT 方向控制设计；翻译通常依赖任务 prefix，而不是 M2M100/NLLB 这种语言 token + `forced_bos_token_id` 机制。
- MADLAD-400 证明 T5 架构可以训练出大规模多语言翻译模型，但它仍是 T5 text-to-text 路线；对本项目来说，核心问题不是某个 checkpoint 的大小，而是 T5 语义不如 M2M100/NLLB 直接匹配 many-to-many MT 的方向控制。
- T5Gemma / T5Gemma2 很新，CTranslate2 也已支持，但它们面向通用、长上下文、多模态或 Gemma 迁移能力，架构复杂度和模型许可边界都比 M2M100 路线更重。
- 本项目第一阶段需要的是可从零训练、可解释、可部署的小型翻译专用基线，而不是引入更通用的新一代 Encoder-Decoder LLM 复杂度。

其他 Encoder-Decoder 取舍：

- Marian/OPUS-MT：CTranslate2 支持成熟，但公开模型以方向模型为主，中英日韩 12 方向覆盖不足。
- mBART / mBART-50：多语言 Encoder-Decoder，可参考，但它的核心路线是大规模多语言单语语料去噪预训练，再进行 MT fine-tune。本项目不会直接微调现成 mBART 权重；如果从零复刻 mBART 路线，反而需要先做去噪预训练再做翻译训练，会增加额外单语数据阶段、总训练量和实验复杂度。根据 SMaLL-100 的经验，小型 MT 模型可以优先走 M2M100 蒸馏/浅 decoder/平行语料训练路线；去噪预训练可作为低资源增强实验，不是 MVP 必要前置。
- NLLB：架构与语言控制方式非常接近本项目，但公开资产是 `CC-BY-NC-4.0`，只能参考设计，不直接使用权重或 tokenizer 文件。
- BART / Pegasus / Whisper：虽属于 Encoder-Decoder 或 seq2seq 系列，但主要面向去噪、摘要或语音任务，不适合作为文本多语言 MT 主线。

### Tokenizer

Tokenizer 从零训练，不使用 Meta NLLB-200 的 `sentencepiece.bpe.model` 或 `tokenizer.json`。项目只复用 NLLB 风格语言 token 控制方式，以及 Transformers 里的 tokenizer 软件实现。

许可证边界：`NllbTokenizerFast` 属于 Hugging Face Transformers 软件代码，Transformers 采用 Apache License 2.0；单纯使用这个类不会让项目自动受 `CC-BY-NC-4.0` 约束。`CC-BY-NC-4.0` 风险来自直接下载或分发 Meta NLLB-200 仓库里的 tokenizer 文件、checkpoint 等资产，例如 `NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")`。

需要固定的运行约定：

- 语言 token：只加入项目需要的 token，例如 `eng_Latn`、`zho_Hans` / `zho_Hant`、`jpn_Jpan`、`kor_Hang`。
- 输入格式：默认采用 NLLB 新行为，encoder 输入为 `<src_lang> source_text </s>`，target/label 为 `<tgt_lang> target_text </s>`。
- 生成控制：推理时使用 `forced_bos_token_id = target_lang_id`；`decoder_start_token_id` 可先沿用 M2M100/NLLB 的 `eos_token_id` 约定，但必须在最小训练和推理脚本中验证。
- 词表一致性：自训练 tokenizer 的 `vocab_size` 必须与模型 embedding 和输出层一致，不能沿用 NLLB-200 的 256k 词表假设。

### 蒸馏路线

训练路线以 Hy-MT2 7B 作为 teacher，以 Diesel-MT 的 M2M100 语义 Encoder-Decoder 作为 student。

- teacher：`tencent/Hy-MT2-7B`，翻译专用、Apache-2.0，作为离线数据生成和质量过滤工具。
- student：从零初始化的小型 Encoder-Decoder，不继承 Hy-MT2 权重、tokenizer 或 decoder-only 架构。
- 目标：用 teacher 生成中英日韩 12 方向的蒸馏平行样本，优先补齐 `zh<->ja`、`zh<->ko`、`ja<->ko` 等公开语料较弱方向。
- 约束：蒸馏提示词必须强制“只输出译文，不要解释”，并记录 teacher 版本、prompt、采样参数、语料来源和过滤规则，保证数据可复现。
- 推理：最终部署只依赖 Diesel-MT student 和 CTranslate2，不依赖 Hy-MT2 teacher。

## 目标基线

目标基线按 `M2M100Config` 字段语义描述。用户描述中的 `header = 12` 在本文中按 `attention heads = 12` 理解。

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
| 目标参数量 | 约 200M |

当前参数估算脚本位于 `scripts/calculate_model_parameters.py`。在共享 Encoder embedding、Decoder embedding 和输出层权重的假设下：

| 模块 | 参数量 |
| --- | ---: |
| Tokenizer/Embedding | 50.33M |
| Encoder | 113.25M |
| Decoder | 37.75M |
| 总计 | 201.33M |

| 精度 | 权重大小 |
| --- | ---: |
| BF16/FP16 | 约 384.0 MiB |
| INT8 | 约 192.0 MiB |
| INT4 理论值 | 约 96.0 MiB |

运行参数估算：

```bash
python scripts/calculate_model_parameters.py
```

## 模型调研摘要

| 模型/项目 | 发布/创建时间 | 许可证 | 关键信息 | 结论 |
| --- | --- | --- | --- | --- |
| [`Helsinki-NLP/opus-mt-*`](https://huggingface.co/Helsinki-NLP) / MarianMT | 2022 | Apache-2.0 / CC-BY-4.0 混合 | CTranslate2 支持成熟；多为单方向模型；已找到 `en<->zh`、`en<->ja`、`en<->ko`、`zh->ja` | `ja->zh`、`zh<->ko`、`ja<->ko` 缺失，12 方向凑不齐 |
| [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M/tree/main) | 2022-03-02，原始工作 2020 | MIT | many-to-many 翻译模型，PyTorch 权重约 1.94 GB | 架构可参考，但权重偏大、模型较老 |
| [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M/tree/main) | 2022-07-08 | CC-BY-NC-4.0 | HF config 使用 `M2M100ForConditionalGeneration` / `model_type = m2m_100`；NLLB 专用 tokenizer；权重约 2.46 GB | 架构和语言 token 设计可参考；不直接使用权重或 tokenizer 资产 |
| [`SMaLL-100`](https://huggingface.co/alirezamsh/small100) | 2022-11-01，论文 2022 | MIT | 小型多语言翻译模型，覆盖 100 语言和 10K+ 语言对；约 333M 参数；需要 `tokenization_small100.py` | 路线可参考，但不直接采用：中英日韩 12 方向需复测，且远端自定义代码增加部署风险 |
| [`facebook/mbart-large-50-many-to-many-mmt`](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) | 2022-03-02，论文 2020 | HF 模型卡/API 未标注 | mBART-large-50 many-to-many MT；基于多语言去噪预训练，覆盖中英日韩，约 0.6B 参数 | 本项目不直接微调现成权重；从零复刻 mBART 会增加去噪预训练阶段和总训练复杂度 |
| [`google-t5/t5-base`](https://huggingface.co/google-t5/t5-base) / T5 | 2020，论文 2019 | Apache-2.0 | 通用 text-to-text Encoder-Decoder；CTranslate2 支持 T5 | 可用于翻译，但不是 many-to-many MT 专用语义，语言方向控制不如 M2M100/NLLB 直接 |
| [`google/t5gemma-*`](https://huggingface.co/docs/transformers/en/model_doc/t5gemma) / [`T5Gemma2`](https://huggingface.co/docs/transformers/en/model_doc/t5gemma2) | T5Gemma 2025，T5Gemma2 2025-12 | Gemma Terms | 新一代 Encoder-Decoder Gemma；CTranslate2 支持 T5Gemma/T5Gemma2；T5Gemma2 覆盖 270M-270M、1B-1B、4B-4B | 新且有参考价值，但偏通用/长上下文/多模态，许可和架构复杂度不适合作为 MVP 主线 |
| [`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt) | 2023-11-27，论文 2023 | Apache-2.0 | T5 架构，400+ 语言，3B 参数；Q4 GGUF 约 1.65 GB | 证明 T5 可做大规模 MT，但 T5 text-to-text 语义不如 M2M100/NLLB 直接匹配本项目方向控制 |
| [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large) | 2023-11-29，论文 2023-12-08 | CC-BY-NC-4.0 | speech/text 多模态翻译，约 2B 参数 | 能力范围过宽，许可证和体积不匹配 |
| [`NiuTrans/LMT-60`](https://huggingface.co/NiuTrans/LMT-60-0.6B) | 2025-11-09，2026-05 更新 | Apache-2.0 | 基于 Qwen3 的 decoder-only 翻译 LLM；60 语言、234 方向；0.6B 页面实际约 0.8B 参数 | 新且许可证友好，但大于 200M 目标，且不是 Encoder-Decoder |
| [`tencent/Hy-MT2-7B`](https://huggingface.co/tencent/Hy-MT2-7B) | 2026-05-21 | Apache-2.0 | Hy-MT2 系列翻译专用 dense teacher；支持 33 语言和多语言翻译指令；HF 显示约 8B 参数 | 不作为本地推理基线；适合作为离线蒸馏 teacher 生成中英日韩 12 方向训练样本 |
| [`tencent/Hy-MT2-1.8B`](https://huggingface.co/tencent/Hy-MT2-1.8B) / [GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/tree/main) / [1.25Bit](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) | 2026-05-21 | Apache-2.0 | 33 语言；1.25Bit 约 440/462 MB；常规 Q4_K_M 约 1.13 GB | 1.25Bit 小体积依赖 STQ 自定义算子；x86 标量回退性能不可作为基线，常规 4bit 后失去大小优势 |
| [`ALMA-7B`](https://huggingface.co/haoranxu/ALMA-7B)、[`TowerInstruct`](https://huggingface.co/Unbabel/TowerInstruct-Mistral-7B-v0.2)、[`Aya-23-8B`](https://huggingface.co/CohereLabs/aya-23-8B)、[`Seed-X`](https://huggingface.co/ByteDance-Seed/Seed-X-Instruct-7B) | 2023-2025 | MIT / CC-BY-NC-4.0 / OpenMDW 等 | 翻译或多语言 instruction LLM，通常 7B/8B | 可参考数据和评估，不作为 CPU/mobile 小模型基线 |

OPUS-MT 方向覆盖摘要：

- 已找到：`en->zh`、`zh->en`、`en->ja`、`ja->en`、`en->ko`、`ko->en`、`zh->ja`。
- 未找到明确公开模型：`ja->zh`、`zh->ko`、`ko->zh`、`ja->ko`、`ko->ja`。

## MVP

MVP 阶段只验证路线，不追求最终效果。

快速验证配置固定 `d_model = 512`、`encoder_ffn_dim = decoder_ffn_dim = 2048`，建议 `attention_heads = 8` 以保持每头 64 维。

| MVP 配置 | `vocab_size` | `encoder_layers` | `decoder_layers` | 参数量 | FP16/BF16 | INT8 | INT4 理论值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e12-d3-v48k` | 48k | 12 | 3 | 75.50M | 144.0 MiB | 72.0 MiB | 36.0 MiB |
| `e12-d3-v32k` | 32k | 12 | 3 | 67.11M | 128.0 MiB | 64.0 MiB | 32.0 MiB |
| `e8-d2-v48k` | 48k | 8 | 2 | 58.72M | 112.0 MiB | 56.0 MiB | 28.0 MiB |
| `e8-d2-v32k` | 32k | 8 | 2 | 50.33M | 96.0 MiB | 48.0 MiB | 24.0 MiB |

验证方式：

- 语料：公开平行语料 + Hy-MT2 7B 蒸馏样本。
- 方向：先跑通 `zh-en`、`en-ja`、`en-ko` 等英语中心方向，再扩展到 12 方向。
- 词表：优先对比 32k 与 48k，观察中日共享汉字、韩文、英文 subword 覆盖和 OOV/碎片率。
- 指标：loss、BLEU、chrF、COMET/参考模型打分、固定样例翻译和失败案例。

验收标准：

- 数据预处理可重复执行。
- 自训练 tokenizer 能稳定编码四种语言，并正确处理语言 token。
- 模型完成一次端到端训练，checkpoint 可恢复训练和执行推理。
- 至少有一组固定测试样例用于回归检查。

## 开发环境

项目默认使用 Python 3.11 和仓库根目录下的 `.conda` 环境。环境创建、激活、验证和依赖安装约定见 [Python 环境约定](docs/python-environment.md)。

## Workflow

项目采用 `plan -> todo -> task -> review -> done` 工作流。README 只说明流程语义，不维护具体任务清单；具体内容应放在 issue、实验记录、任务文档或代码提交中。

### plan

说明当前阶段要解决的问题、边界、约束和验收标准。plan 用于统一方向，不承载具体执行列表。

### todo

把 plan 拆成可执行的待办项。todo 应放在项目管理工具或独立任务文件中，README 不记录具体条目。

### task

将 todo 拆成可独立完成、验证和提交的工作单元。每个 task 应有输入、输出、验证方式和记录位置。

### review

对 task 的结果进行复核，重点检查可复现性、正确性、数据边界、评估结果和部署风险。review 通过后才能进入 done。

### done

表示 task 已完成实现、验证和记录。done 状态应能追溯到代码、实验结果或评审记录，而不是只停留在口头结论。

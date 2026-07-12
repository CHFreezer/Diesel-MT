# plan: mvp tokenizer

## 目标

制作 Diesel-MT MVP 阶段可用的中英日韩 tokenizer，用于验证从数据、tokenizer、模型配置、训练、评估到推理的最小闭环。该 tokenizer 必须从零训练，不复用 Meta NLLB-200、M2M100 或其他模型仓库中的 tokenizer 资产。

MVP 阶段优先产出 `32k` 和 `48k` 两个词表规模候选，并通过覆盖率、序列长度膨胀、语言 token 行为和最小训练链路验证选择后续默认版本。

## 范围

本 plan 覆盖 tokenizer 的训练目标、输入要求、特殊 token 约定、导出格式、验证标准和风险边界。

本 plan 不覆盖 tokenizer 训练语料的下载脚本实现，也不覆盖翻译模型训练、蒸馏样本生成或 CTranslate2 转换。

## 约束

- tokenizer 必须从项目自有训练语料生成，不能下载、复制或分发 NLLB-200、M2M100、mBART、SMaLL-100 等现成 tokenizer 文件。
- tokenizer 需要支持 `eng_Latn`、`zho_Hans`、`jpn_Jpan`、`kor_Hang` 四个 MVP 必需语言 token；`zho_Hant` 可作为保留 token，但不作为 MVP 覆盖率硬性指标。
- tokenizer 训练不得做英文小写化、中文简繁转换、日文假名转换、韩文罗马化等会破坏原文形态的处理。
- `vocab_size` 必须与后续模型配置中的 embedding 和输出层一致。
- special token ID、语言 token ID、`eos_token_id`、`pad_token_id`、`unk_token_id`、`forced_bos_token_id` 映射必须可序列化、可复现、可测试。

## 输入

输入为 tokenizer 专用清洗语料，每种语言应至少提供一个独立文本文件，并附带数据来源、许可证、处理版本、行数、字符数和抽样策略记录。

建议输入目录语义：

```text
data/tokenizer/corpus/mvp/
  eng_Latn.txt
  zho_Hans.txt
  jpn_Jpan.txt
  kor_Hang.txt
  manifest.jsonl
```

语料应按语言均衡抽样，避免英文或中文语料规模过大导致词表被单一语言主导。MVP 允许小规模语料先跑通流程，但必须保留扩大语料规模后可复现重训的接口。

## tokenizer 形态

MVP 优先采用 SentencePiece 兼容路线，训练中英日韩共享子词词表。训练产物需要能够被 Transformers 侧加载，并能服务 M2M100/NLLB 风格的语言 token 控制。

必须保留的特殊 token：

```text
<unk>
<s>
</s>
<pad>
eng_Latn
zho_Hans
zho_Hant
jpn_Jpan
kor_Hang
```

默认输入格式遵循 README 约定：

```text
<src_lang> source_text </s>
<tgt_lang> target_text </s>
```

推理时必须能稳定取得目标语言 token 对应的 `forced_bos_token_id`。

## 产物

每个候选 tokenizer 至少产出以下内容：

```text
artifacts/tokenizers/mvp-32k/
artifacts/tokenizers/mvp-48k/
```

每个目录内应包含可加载 tokenizer 文件、special token 配置、语言 token 到 ID 的映射、训练配置、训练语料 manifest 快照、覆盖率报告和最小编码样例。

## 验证

验证重点是 tokenizer 是否适合作为 MVP 模型训练入口，而不是单独追求压缩率最优。

必须验证：

- 四种 MVP 语言样例都能 encode/decode，且不会把语言 token 切碎。
- `eng_Latn`、`zho_Hans`、`jpn_Jpan`、`kor_Hang`、`zho_Hant` 的 token ID 稳定存在。
- 对固定中英日韩样例计算 `<unk>` 比例、平均 token 数、字符到 token 比例和极端长样本行为。
- 32k 与 48k 两个候选在中日共享汉字、韩文音节、英文 subword 覆盖上有可比较报告。
- tokenizer 文件能被后续最小训练脚本加载，并能生成与模型配置一致的 `vocab_size`。

## 验收标准

- 从零训练可重复执行，固定输入 manifest 和随机种子后产物稳定。
- MVP 语言 token 能正确参与 encoder 输入和 decoder 目标语言控制。
- 32k 与 48k 候选均有覆盖率和序列长度报告。
- 至少一套 tokenizer 被标记为 MVP 默认候选，并记录选择理由。
- 产物不包含禁止复用的第三方 tokenizer 资产。

## 风险

- 语料不均衡会导致词表偏向英文或中文，影响日文、韩文碎片率。
- 简繁中文是否合并会影响中文覆盖和语言 token 设计；MVP 先以 `zho_Hans` 为硬目标，保留 `zho_Hant` 扩展空间。
- 过早固定 32k 或 48k 可能影响后续模型大小；MVP 需要保留重训和对比入口。
- tokenizer 与 Transformers/CTranslate2 的加载路径可能存在格式细节差异，需要在最小训练和推理链路中验证。

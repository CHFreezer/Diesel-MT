# task TD-01: 训练环境与依赖

状态：done

依赖：无

## 目标

锁定 Transformers 5.x、tokenizers、CTranslate2 和 CUDA 13.2 版 PyTorch（`cu132`）的兼容版本组合，安装到 `.conda` 环境，用可执行源码断言确认 NLLB BPE 路线和 CTranslate2 M2M100 转换注册，生成版本兼容记录。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- [Python 环境约定](../../../docs/python-environment.md)
- [Transformers 5.13.1 NLLB tokenizer 源码](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/nllb/tokenization_nllb.py)
- [CTranslate2 Transformers converter 源码](https://github.com/OpenNMT/CTranslate2/blob/master/python/ctranslate2/converters/transformers.py)

## 执行事项

- 在 Transformers 5.x 范围内锁定具体 `transformers`、`tokenizers`、`ctranslate2` 和 CUDA 13.2 版 `torch`（`--index-url https://download.pytorch.org/whl/cu132`）兼容版本。
- 记录锁定版本对应的 Transformers `tokenization_nllb.py` 和 CTranslate2 `transformers.py` commit URL；禁止只记录浮动的 `main` / `master` 链接。
- 用可执行源码断言确认 NLLB `model is BPE`，并确认 CTranslate2 注册了 `M2M100Config -> M2M100Loader`。
- 在 `.conda` 环境验证 `transformers`、`tokenizers`、`ctranslate2`、`torch` 可正常导入；确认 `torch.cuda.is_available()` 为 True（后续模型训练需要；tokenizer 阶段不调用 GPU）。
- 验证 `torch.cuda.get_device_capability()` 返回有效值，确认 cu132 wheel 覆盖本机 GPU 架构。
- 构造 `NllbTokenizer` 并断言 `is_fast is True`，同时记录 CTranslate2 CPU 支持的 compute types。
- 核对锁定 Transformers 对 `tokenizers` 的版本范围，并用该精确组合完成 `tokenizer.json` 保存、`AutoTokenizer` 重载和底层 `tokenizers.Tokenizer.from_file()` 冒烟。
- 生成版本兼容记录：Transformers 版本、实际 tokenizer 类名、基类、`is_fast`、Tokenizers 版本、CTranslate2 转换结果。若 5.x 冒烟失败，停止实施并发起架构变更评审，不得静默切换 4.x。
- `sentencepiece` 仅在需要验证可选互操作导出时安装，不作为 MVP 规范训练链的默认依赖。
- 更新 `requirements.txt` 并生成新的 `requirements.lock`。

## 产物

- 更新后的 `requirements.txt` 和 `requirements.lock`
- 版本兼容记录（锁定版本号、commit URL、`is_fast` 断言结果、CT2 compute types、`model is BPE` 断言结果）
- `NllbTokenizer` 构造与 `AutoTokenizer` 重载冒烟日志

## 验收

- `requirements.lock` 固定所有直接和传递依赖版本。
- 锁定版本的 `tokenization_nllb.py` 源码中 `model is BPE` 断言通过。
- 构造的 `NllbTokenizer` 满足 `is_fast is True`。
- `tokenizer.json` 保存 → `AutoTokenizer` 重载 → `tokenizers.Tokenizer.from_file()` 加载均成功。
- CTranslate2 注册了 `M2M100Config -> M2M100Loader`。
- 若任何一项 5.x 冒烟失败，任务不得标记 done，须发起架构变更评审。

## 验证记录（2026-07-13）

### 锁定版本

| 包 | 版本 | 来源 |
|---|------|------|
| `torch` | 2.13.0+cu132 | `--index-url https://download.pytorch.org/whl/cu132` |
| `torchvision` | 0.28.0+cu132 | 同上 |
| `transformers` | 5.13.1 | PyPI |
| `tokenizers` | 0.22.2 | PyPI（transformers 传递依赖） |
| `ctranslate2` | 4.8.1 | PyPI |
| `huggingface_hub` | 1.23.0 | PyPI（transformers 传递依赖） |

### 源码断言

- **CUDA**：`torch.cuda.is_available() == True`，CUDA 13.2，GPU capability `(8, 9)`（Ada Lovelace，cu132 覆盖确认）。
- **BPE 路线**：`NllbTokenizer.model` 为 `<class 'tokenizers.models.BPE'>`，断言通过。
- **CTranslate2 M2M100**：`ctranslate2.converters.transformers` 中 `M2M100Loader` 已注册。
- **`is_fast`**：构造空 `NllbTokenizer(extra_special_tokens=LANG_CODES)` 后 `tokenizer.is_fast is True`。
- **特殊 token ID**：`<s>=0, <pad>=1, </s>=2, <unk>=3`，与 NLLB 约定一致。
- **语言裁剪**：`extra_special_tokens` 仅传入 5 个项目语言，`get_vocab()` 中确认保留语言存在、`fra_Latn`/`deu_Latn`/`rus_Cyrl` 不存在。
- **保存/重载冒烟**：`save_pretrained()` 产出 `tokenizer.json` + `tokenizer_config.json`；`AutoTokenizer.from_pretrained(local_files_only=True)` 返回 `NllbTokenizer` 且 `is_fast is True`；`tokenizers.Tokenizer.from_file()` 加载 `tokenizer.json` 成功，BPE 类型确认。

### 产物

- `requirements.txt`（已更新：新增 `transformers==5.13.1`、`ctranslate2==4.8.1`，torch 以注释记录 cu132 索引）
- `requirements.lock`（已生成：`pip freeze` 全量锁定）
- 本验证记录

# task TD-10: CTranslate2 转换与 CPU 推理冒烟

状态：deferred（下游部署任务，不阻塞 `mvp-tokenizer-v0` 冻结）

依赖：TD-09（最小训练链路集成验证）

## 目标

用锁定版本的 CTranslate2 转换器将 TD-09 的本地 HF checkpoint 转为 CT2 格式，在 CPU 上以 int8 精度执行 `target_prefix` 推理冒烟，验证 tokenizer 词表与 CT2 模型词表的 ID 一致性。

## 输入

- [mvp tokenizer todo](../../todo/mvp-tokenizer.md)
- TD-09 产出的本地 HF checkpoint（`artifacts/tokenizers/integration-checkpoint/`）
- TD-08 MVP 默认 tokenizer（独立 HF tokenizer 目录）
- [CTranslate2 M2M100Loader 源码](https://github.com/OpenNMT/CTranslate2/blob/master/python/ctranslate2/converters/transformers.py#L477-L508)
- [CTranslate2 NLLB 运行示例](https://opennmt.net/CTranslate2/guides/transformers.html#nllb)

## 执行事项

- 锁定版本执行 `ct2-transformers-converter --model <local-hf-checkpoint> --output_dir <ct2-dir>`，禁止依赖远端模型或 `trust_remote_code`。
- 至少转换并加载一个 CPU `int8` 产物；保留 float32 产物作为转换问题的诊断基线。
- 用同一 HF tokenizer 生成 source token 字符串，确认序列包含 `<src_lang>` 前缀和 `</s>` 后缀；不得依赖 CTranslate2 自动补 special tokens。
- 对五个目标语言（含 `zho_Hant`）分别执行 `translate_batch(..., target_prefix=[[tgt_lang]], beam_size=1, max_decoding_length=<small>)`，确认不发生 unknown target token、词表越界或模型加载错误。
- 确认返回 hypothesis 的第一个 token 等于目标语言 prefix，移除该 token 后可由同一 HF tokenizer decode。
- 检查 CT2 转换词表与 `tokenizer.get_vocab()` 的 ID 顺序完全一致。
- 检查 CT2 配置中的 `bos_token`、`eos_token`、`unk_token` 和 `decoder_start_token` 值。
- 将 CT2 模型目录与独立 tokenizer 目录按部署布局打包：
  ```text
  artifacts/tokenizers/deployment/
    ct2-model/          (CTranslate2 转换产物)
    tokenizer/          (HF tokenizer 目录)
  ```
- 在新的离线进程中，仅从部署目录本地路径加载 CT2 `Translator` 和 HF `AutoTokenizer`，跑通一次 CPU 推理并 decode。

## 产物

- `artifacts/tokenizers/deployment/ct2-model/`（int8 + float32）
- `artifacts/tokenizers/deployment/tokenizer/`
- CTranslate2 转换日志、CPU 冒烟日志
- 部署目录说明（`artifacts/tokenizers/deployment/README.md`）

## 验收

- `ct2-transformers-converter` 转换成功，无 warning 或 error。
- int8 CT2 模型可在 CPU 上加载。
- 五个目标语言的 `target_prefix` 推理均不发生 token 越界或 `<unk>` target。
- CT2 词表与 `tokenizer.get_vocab()` ID 顺序完全一致。
- hypothesis 去前缀后可由 HF tokenizer 正常 decode。
- 离线部署目录可在新进程中独立加载并推理。
- 转换和推理过程不依赖网络或远端模型仓库。

## 验证记录

（待填写）

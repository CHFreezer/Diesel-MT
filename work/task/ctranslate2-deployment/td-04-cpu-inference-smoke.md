# task TD-04: 五语言 CPU 推理冒烟

状态：pending

依赖：TD-03

## 目标

使用同一 HF tokenizer 和 CTranslate2 `Translator`，验证五个目标语言的 source token、`target_prefix`、CPU 推理、去 prefix 和 decode 接口闭环。

## 输入

- [CTranslate2 deployment todo](../../todo/ctranslate2-deployment.md)
- TD-02 转换产物与 TD-03 完整性报告

## 执行事项

- 为每个源语言构造本地固定样例，并由 HF tokenizer 生成 token 字符串。
- 断言 source token 首项为源语言 token、末项为 `</s>`，不由 CT2 隐式补齐。
- 对五个目标语言调用 `translate_batch(..., target_prefix=[[tgt_lang]], beam_size=1)`，限制短解码长度。
- 断言 hypothesis 首 token 为目标语言 prefix，移除后用同一 tokenizer decode。
- 对 float32 和 int8 至少各执行一轮，记录 compute type、耗时和异常。

## 验收

- 五个目标语言均无 unknown target token、词表越界、模型加载或 decode 错误。
- float32 与 int8 接口行为均通过；不对随机输出的翻译语义作质量断言。
- 固定输入、token 序列、prefix、输出 token 和日志可追溯。

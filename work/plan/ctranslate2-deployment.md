# plan: CTranslate2 deployment validation

状态：done / archived

## 来源

- 已冻结 tokenizer plan：[mvp tokenizer](mvp-tokenizer.md)
- 已归档 tokenizer todo：[mvp tokenizer](../done/todo/mvp-tokenizer.md)
- 延期任务记录：[TD-10 CTranslate2](../done/task/mvp-tokenizer/td-10-ctranslate2-smoke.md)
- 冻结验收：[mvp-tokenizer-v0](../../artifacts/tokenizers/reports/mvp-tokenizer-v0/freeze_acceptance.md)

## 目标

使用不可变的五标签 `mvp-tokenizer-v0` 和随机初始化的微型 `M2M100ForConditionalGeneration` checkpoint，验证 Hugging Face checkpoint 到 CTranslate2 的转换、词表/ID 空间一致性、CPU `int8` 加载、五标签 `target_prefix` 推理、decode 和离线部署目录闭环。

本 plan 承接 tokenizer 工作流中明确延期的 CTranslate2 事项，不重新训练、修改或重新冻结 tokenizer。

## 范围

包含：

- 生成可重复、可离线重建的微型 HF checkpoint；
- 生成 CTranslate2 float32 诊断产物和 CPU int8 验收产物；
- 核对 tokenizer、HF embedding/lm_head、CT2 词表和特殊 token；
- 对五个目标语言执行 CPU 推理接口冒烟；
- 验证 tokenizer 与 CT2 模型分离的离线部署布局；
- 增加自动化慢速集成测试和可追溯运行记录。

不包含：

- 正式翻译模型训练、蒸馏或质量评估；
- 随机微型模型输出的语义正确性判断；
- 生产级吞吐、延迟、量化精度或服务容量调优；
- 修改 `mvp-tokenizer-v0` 的词表、token ID 或冻结文件。

## 不变量

- tokenizer 必须从 `artifacts/tokenizers/mvp-tokenizer-v0/` 本地加载，冻结根保持 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。
- HF `config.vocab_size`、shared/encoder/decoder embedding、`lm_head` 和 CT2 词表均为 49,152 行/项。
- source token 必须由同一 HF tokenizer 生成并包含 `<src_lang>` 前缀与 `</s>` 后缀；不得假设 CTranslate2 自动补齐。
- 目标语言通过 `target_prefix=[[tgt_lang]]` 指定；返回结果移除首个语言 prefix 后再交给同一 tokenizer decode。
- 转换和推理必须完全使用本地文件，不依赖远端模型、`trust_remote_code` 或运行时下载。
- 随机模型只用于接口和 ID 空间验证，不作为可发布翻译模型。

## 产物

- 可重建的微型 HF checkpoint 配置、生成脚本或测试 fixture；
- float32 与 CPU int8 CTranslate2 转换记录；
- 词表、特殊 token、模型维度和文件哈希校验报告；
- 五标签 CPU 推理冒烟日志；
- `tokenizer/` 与 `model/` 分离的离线部署布局说明；
- 自动化慢速集成测试。

大体积随机权重和转换目录默认视为可再生运行产物，不直接提交 Git；仓库提交生成逻辑、配置、manifest、测试和必要的精简日志。

## 验收标准

- 微型 HF checkpoint 可从锁定依赖和固定种子重复生成并离线重载。
- float32 转换成功，可作为 int8 问题的诊断基线；CPU int8 模型转换并加载成功。
- tokenizer/HF/CT2 的完整 ID 空间一致，五个语言 token、`<s>`、`<pad>`、`</s>`、`<unk>` 无缺失或错位。
- 五个目标语言均能完成 source tokenize、`target_prefix` 推理、去 prefix 和 decode，无词表越界或未知目标 token。
- 在新离线进程中仅依赖部署目录即可完成一次 CPU 冒烟。
- 自动化测试通过，命令、版本、哈希、已知限制和运行日志齐全。

## 执行拆解

- todo：[CTranslate2 deployment validation](../done/todo/ctranslate2-deployment.md)
- task 索引：[CTranslate2 deployment tasks](../done/task/ctranslate2-deployment/index.md)

## 完成记录

- 统一 review：[CTranslate2 deployment validation review](../done/review/ctranslate2-deployment.md)
- 合并机器记录：[deployment-validation.json](../../artifacts/ctranslate2/deployment-validation.json)
- 最终自动化回归：`52 passed`
- 冻结 tokenizer manifest SHA-256：`eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`
- 有序词表 SHA-256：`72bc2edcfe44bdfac90d2c101f71f214cdd2c4b70d7c975e5d269011be40c716`
- INT8 离线部署包 manifest SHA-256：`a19a189fb1288cec8d2d232601086b4b1fb0dc4a80902cd7a0f478ce301c5c86`

## 完成边界

本 plan 完成只证明冻结 tokenizer 与 CTranslate2/M2M100 部署接口兼容。正式模型可用性仍须由后续模型训练、翻译质量和性能工作流单独验收。

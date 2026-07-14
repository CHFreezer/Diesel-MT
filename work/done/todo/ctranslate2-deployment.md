# todo: CTranslate2 deployment validation

状态：done

## 来源

- plan：[CTranslate2 deployment validation](../../plan/ctranslate2-deployment.md)
- task 索引：[CTranslate2 deployment tasks](../task/ctranslate2-deployment/index.md)
- 合并机器记录：[deployment-validation.json](../../../artifacts/ctranslate2/deployment-validation.json)
- 已归档 tokenizer todo：[mvp tokenizer](mvp-tokenizer.md)
- tokenizer review：[mvp tokenizer review](../review/mvp-tokenizer.md)

## 目标

承接 `mvp-tokenizer-v0` 冻结工作流中的 CTranslate2 延期项，完成微型 HF checkpoint、CT2 转换、词表完整性、五语言 CPU 推理和离线部署验证。

## 待办

### TD-01 微型 HF checkpoint

- [x] 固定随机种子和最小 `M2M100Config`，将 `vocab_size` 与 tokenizer 的 49,152 ID 空间绑定。
- [x] 生成并离线重载微型 HF checkpoint，核对 shared/encoder/decoder embedding 和 `lm_head` 维度。
- [x] 保存生成配置、依赖版本、文件清单和 SHA-256；明确随机权重不用于翻译质量判断。

### TD-02 CTranslate2 转换

- [x] 使用锁定版本 `ct2-transformers-converter` 从本地 checkpoint 生成 float32 诊断产物。
- [x] 生成 CPU int8 验收产物，记录完整命令、环境、耗时、输出文件和哈希。
- [x] 验证转换过程无网络访问、远端模型依赖或 `trust_remote_code`。

### TD-03 词表与 ID 空间完整性

- [x] 逐 ID 对比 HF tokenizer 词表、HF 模型维度和 CT2 转换词表。
- [x] 核对五个 language token 与 `<s>`、`<pad>`、`</s>`、`<unk>` 的 token/ID/config 语义。
- [x] 对重复 token、ID 空洞、词表长度差异或配置错位明确失败。

### TD-04 五语言 CPU 推理冒烟

- [x] 用同一 HF tokenizer 生成含 source language token 和 `</s>` 的 source tokens。
- [x] 对五个目标语言执行 `target_prefix`、beam size 1 的短序列 CPU 推理。
- [x] 验证 hypothesis 首 token 等于目标语言 prefix，移除后可由同一 tokenizer decode。
- [x] 分别验证 float32 与 int8 至少一次，记录 compute type 和运行日志。

### TD-05 离线部署包与自动化测试

- [x] 定义独立 `tokenizer/` 与 `model/` 的部署布局，并生成文件 manifest。
- [x] 在新离线进程中只从部署目录加载并完成一次端到端 CPU 冒烟。
- [x] 增加可重复运行的慢速集成测试，覆盖转换、词表一致性、五语言 prefix 和 decode。
- [x] 更新部署说明、已知限制和验收记录。

## 完成条件

- [x] TD-01 至 TD-05 全部通过并有可追溯验证记录。
- [x] tokenizer 冻结根未变化，未修改 `mvp-tokenizer-v0`。
- [x] float32 和 CPU int8 转换/加载通过，五个目标语言推理接口冒烟通过。
- [x] 离线部署布局和自动化测试通过。
- [x] 文档明确本结果不代表随机模型具备翻译质量。

## 统一 review 与归档

- [x] 对本 todo 和 TD-01 至 TD-05 的完整产物执行一次[统一 review](../review/ctranslate2-deployment.md)。
- [x] review 通过后，将 todo、task 集合和 review 记录一起归档到 `work/done/`。

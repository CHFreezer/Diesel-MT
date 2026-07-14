# todo: CTranslate2 deployment validation

状态：pending

## 来源

- plan：[CTranslate2 deployment validation](../plan/ctranslate2-deployment.md)
- task 索引：[CTranslate2 deployment tasks](../task/ctranslate2-deployment/index.md)
- 已归档 tokenizer todo：[mvp tokenizer](../done/todo/mvp-tokenizer.md)
- tokenizer review：[mvp tokenizer review](../done/review/mvp-tokenizer.md)

## 目标

承接 `mvp-tokenizer-v0` 冻结工作流中的 CTranslate2 延期项，完成微型 HF checkpoint、CT2 转换、词表完整性、五语言 CPU 推理和离线部署验证。

## 待办

### TD-01 微型 HF checkpoint

- [ ] 固定随机种子和最小 `M2M100Config`，将 `vocab_size` 与 tokenizer 的 49,152 ID 空间绑定。
- [ ] 生成并离线重载微型 HF checkpoint，核对 shared/encoder/decoder embedding 和 `lm_head` 维度。
- [ ] 保存生成配置、依赖版本、文件清单和 SHA-256；明确随机权重不用于翻译质量判断。

### TD-02 CTranslate2 转换

- [ ] 使用锁定版本 `ct2-transformers-converter` 从本地 checkpoint 生成 float32 诊断产物。
- [ ] 生成 CPU int8 验收产物，记录完整命令、环境、耗时、输出文件和哈希。
- [ ] 验证转换过程无网络访问、远端模型依赖或 `trust_remote_code`。

### TD-03 词表与 ID 空间完整性

- [ ] 逐 ID 对比 HF tokenizer 词表、HF 模型维度和 CT2 转换词表。
- [ ] 核对五个 language token 与 `<s>`、`<pad>`、`</s>`、`<unk>` 的 token/ID/config 语义。
- [ ] 对重复 token、ID 空洞、词表长度差异或配置错位明确失败。

### TD-04 五语言 CPU 推理冒烟

- [ ] 用同一 HF tokenizer 生成含 source language token 和 `</s>` 的 source tokens。
- [ ] 对五个目标语言执行 `target_prefix`、beam size 1 的短序列 CPU 推理。
- [ ] 验证 hypothesis 首 token 等于目标语言 prefix，移除后可由同一 tokenizer decode。
- [ ] 分别验证 float32 与 int8 至少一次，记录 compute type 和运行日志。

### TD-05 离线部署包与自动化测试

- [ ] 定义独立 `tokenizer/` 与 `model/` 的部署布局，并生成文件 manifest。
- [ ] 在新离线进程中只从部署目录加载并完成一次端到端 CPU 冒烟。
- [ ] 增加可重复运行的慢速集成测试，覆盖转换、词表一致性、五语言 prefix 和 decode。
- [ ] 更新部署说明、已知限制和验收记录，完成 review 后归档本 todo/task。

## 完成条件

- [ ] TD-01 至 TD-05 全部通过并有可追溯验证记录。
- [ ] tokenizer 冻结根未变化，未修改 `mvp-tokenizer-v0`。
- [ ] float32 和 CPU int8 转换/加载通过，五个目标语言推理接口冒烟通过。
- [ ] 离线部署布局和自动化测试通过。
- [ ] 文档明确本结果不代表随机模型具备翻译质量。

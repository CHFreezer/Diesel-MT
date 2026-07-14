# task TD-06: 锁定并验证 Hy-MT2 7B teacher 运行时

状态：pending

依赖：TD-01

## 目标

锁定官方 Hy-MT2 7B 的模型、代码、许可证与可执行 artifact 身份，在受控环境中建立可完全离线重载、适配本机资源且覆盖五项目标签的 teacher 运行 profile。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01 artifact、路径与 Git 边界
- 官方 [`tencent/Hy-MT2-7B`](https://huggingface.co/tencent/Hy-MT2-7B) 模型卡与 [Apache-2.0 许可证](https://huggingface.co/tencent/Hy-MT2-7B/blob/main/LICENSE.txt)
- RTX 4060 Ti 16 GB、CPU/RAM 与可用 SSD staging

## 原子边界

本 task 只验证并锁定 teacher runtime，不校准最终 prompt/decode、不批量生成训练数据，也不允许 teacher 依赖改写 student 主环境。

## 执行事项

- 锁定官方模型或经验证的官方同模型运行 artifact，记录 revision、模型/代码/chat template/许可证文件清单、大小和 SHA-256。
- 记录 Apache-2.0 模型许可证，并明确其不自动解决输入语料或生成数据的权利边界。
- 审查并锁定 `trust_remote_code` 内容；正式运行从本地固定快照加载，启用离线和网络阻断，不执行浮动 `main`。
- 建立与 student 隔离或明确兼容的 teacher profile，锁定 Python、Transformers、PyTorch、CUDA/后端和命令。
- 在 RTX 4060 Ti/CPU 上比较官方 BF16 offload、FP8、GGUF 等可行路径；只接受官方来源且通过参考集验证的 artifact。
- 对 `zho_Hans`、`zho_Hant`、`eng_Latn`、`jpn_Jpan`、`kor_Hang` 完成最小离线推理，确保输出非空。
- 记录峰值 RAM/VRAM、延迟、吞吐、输出稳定性和限制；无可接受路径则阻塞 D0，不替换 teacher。

## 产物

- teacher artifact lock 与逐文件哈希。
- remote-code/chat-template 审查记录。
- 隔离的离线运行 profile、五标签冒烟和资源报告。

## 验收

- 固定 artifact 可在网络阻断下重载并执行五标签推理。
- 所有执行代码、依赖、模型和许可证身份可审计。
- 选定运行路径通过参考输出一致性检查且适合受控生成。
- 未使用来源不明的社区量化或浮动远端代码。

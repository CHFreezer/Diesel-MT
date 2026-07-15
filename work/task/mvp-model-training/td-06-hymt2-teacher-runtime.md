# task TD-06: 锁定并验证 Hy-MT2 7B teacher 运行时

状态：completed

依赖：TD-01

## 目标

锁定官方 Hy-MT2 7B 的模型、代码、许可证与可执行 artifact 身份，在受控环境中建立可完全离线重载、适配本机资源且覆盖五项目标签的 teacher 运行 profile。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01 artifact、路径与 Git 边界
- 官方 [`tencent/Hy-MT2-7B`](https://huggingface.co/tencent/Hy-MT2-7B) 模型卡与 [Apache-2.0 许可证](https://huggingface.co/tencent/Hy-MT2-7B/blob/main/LICENSE.txt)
- 当前执行主机的 accelerator、CPU/RAM 与可配置 staging；实际硬件身份写入运行证据

## 原子边界

本 task 只验证并锁定 teacher runtime，不校准最终 prompt/decode、不批量生成训练数据，也不允许 teacher 依赖改写 student 主环境。

## 执行事项

- 锁定官方模型或经验证的官方同模型运行 artifact，记录 revision、模型/代码/chat template/许可证文件清单、大小和 SHA-256。
- 记录 Apache-2.0 模型许可证，并明确其不自动解决输入语料或生成数据的权利边界。
- 审查并锁定 `trust_remote_code` 内容；正式运行从本地固定快照加载，启用离线和网络阻断，不执行浮动 `main`。
- 建立与 student 隔离或明确兼容的 teacher profile，锁定 Python、Transformers、PyTorch、CUDA/后端和命令。
- 在当前执行主机的 accelerator/CPU 上比较官方 BF16 offload、FP8、GGUF 等可行路径；只接受官方来源且通过参考集验证的 artifact。
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

## 阶段记录

2026-07-15 先完成官方 FP8 路径的阶段性验收，随后按新增测评要求补充官方 BF16 + bitsandbytes LLM.int8 与官方 GGUF Q8_0 + CUDA 的同口径对比：

- 锁定官方 `tencent/Hy-MT2-7B-FP8` revision `883d09eb21d9be92058556cd0a4016d8a648c7db`，14 个顶层文件共 8,046,445,711 bytes，全部逐文件 SHA-256 验证通过。
- 官方 `dtype=torch.bfloat16`、`device_map="auto"`、`trust_remote_code=True` 示例原样运行通过；快照无 Python/`auto_map`，正式离线 profile 使用等价的原生 Transformers 实现并关闭 remote trust。
- teacher 专用 overlay 未改写 student `.conda`；完全离线运行同时启用 Hugging Face offline 标志与 Python socket 阻断。
- `eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang` 各执行 2 次，10/10 非空、重复一致、阶段性 smoke 输出一致；socket 尝试为 0。FP8 输出后来明确不作为质量基线。
- M0 最大 189 字符 train source 容量探针通过。最终一次运行峰值 CUDA allocated/reserved 为 15,185,548,800 / 16,743,661,568 bytes，短探针平均 3.79 tokens/s。
- 已明确记录：当前 Transformers/`compressed-tensors` 在首次 forward 把 FP8 解压为 BF16，因此只验收 batch=1、M0 短文本、独占 GPU 的受控路径；不宣称硬件 FP8。真正 W8A8 FP8 需要非原生 Windows 的 vLLM/WSL 路径，留待单独系统决策。

产物为 `configs/hymt2_teacher_artifact.lock.json`、`configs/hymt2_teacher_runtime.yaml`、`requirements-teacher-hymt2.txt`、三个 teacher runtime 脚本、`docs/hymt2-teacher-runtime.md` 和 `artifacts/model-training/hymt2-teacher-runtime.json`。

### 原版 BF16、INT8 与 GGUF 三方对比及最终选型

按 2026-07-15 新增要求，下载并逐文件验证官方 `tencent/Hy-MT2-7B` revision `9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223` 与官方 `tencent/Hy-MT2-7B-GGUF` revision `ab8472660ac61fac25f1af43fac2599d52a8a775` 的 `HY-MT2-7B-Q8_0.gguf`。三方分别为原版未量化 `torch.bfloat16` + `device_map=auto`、`bitsandbytes==0.49.2` + `BNB_CUDA_VERSION=130` + LLM.int8，以及官方 llama.cpp `b10012` CUDA 13.3 + GGUF Q8_0。

在同一组五标签 prompt、greedy 解码、每标签 2 次、200 ms 资源采样下：

- 原版 BF16：平均 4.17 tokens/s，峰值显存增量 14,543 MiB，峰值进程 RSS 17,738,473,472 bytes；16 GB 显存无法全驻留，最后 3 层及 norm/rotary 由 `device_map=auto` 卸载到 CPU。
- bitsandbytes LLM.int8：实际加载 `libbitsandbytes_cuda130.dll`，平均 8.79 tokens/s，峰值显存增量 9,687 MiB，峰值进程 RSS 17,030,828,032 bytes，无 CPU/disk offload。
- GGUF Q8_0：平均 27.71 tokens/s，峰值显存增量 7,909 MiB，峰值进程 RSS 8,462,090,240 bytes，全层 CUDA offload。
- GGUF 生成速度为原版 BF16 的 6.65 倍、INT8 的 3.15 倍，峰值显存分别少 6,634 / 1,778 MiB。
- 质量基线改为原版 BF16：INT8 与 GGUF 的五标签短探针均 10/10 逐字匹配 BF16。189 字符容量探针中 INT8 与 BF16 逐字一致，GGUF 有一处 `加熱`/`調理` 措辞差异；三者均触及 64-token 诊断上限，该差异交由 TD-07 在冻结人类 reference 上判断。

锁与机器证据见 `configs/hymt2_teacher_benchmark.lock.json` 和 `artifacts/model-training/hymt2-teacher-runtime-comparison.json`。

最终冻结官方 `tencent/Hy-MT2-7B-GGUF` Q8_0 为 sequence-level 蒸馏源：revision `ab8472660ac61fac25f1af43fac2599d52a8a775`、文件 `HY-MT2-7B-Q8_0.gguf`、SHA-256 `58b3ad55dd6f6fa08c695cddc34fb5f8f708a844f78ae10508071914b0ed67c0`、llama.cpp `b10012` CUDA 13.3。规范入口为 `configs/hymt2_teacher_selection.yaml`；原版 BF16 是后续量化输出比较基线，FP8 不再承担质量参考角色。

TD-07 负责五标签语言名称、逐路由 prompt/decode、人类 reference 质量与相对原版 BF16 的量化差异校准。18 路 v1 校准已完成；新增简繁互转两路继续沿用 `Chinese` / `Traditional Chinese` 名称。若发现不可接受退化则阻塞 addendum，不得静默回退后端；TD-06 的 artifact/runtime 状态保持完成。

### 本地 runtime 归位与清理

按最终存储决策，将选定 GGUF Q8_0、llama.cpp CUDA `bin/`、原版 BF16 快照、BF16/INT8 隔离 overlay 和原始 JSON 证据迁入工作目录下 Git-ignored 的 `artifacts/model-training/runtime/`。原版权重目录命名为 `teacher/hymt2-7b-bf16/`；bitsandbytes INT8 复用该 BF16 快照并在加载时动态量化，不另存一套权重。原版 BF16 与 GGUF 锁定文件均在目标位置完成逐文件 SHA-256 验证，迁移后的 overlay 使用 `BNB_CUDA_VERSION=130` 成功加载 `libbitsandbytes_cuda130.dll`。

从工作目录实际加载验证通过：原版 BF16 载入 5.40 秒并生成 `The weather is nice today.`；GGUF + llama.cpp CUDA server 载入 3.52 秒并生成相同译文。该 HDD runtime 只用于顺序加载到 RAM/VRAM 和低频只读访问，不承载热 checkpoint、随机写缓存或频繁日志。

已删除 D 盘旧 runtime、未选中 FP8 权重、Hugging Face `.cache`、llama.cpp 下载压缩包、GGUF 测试 venv 和 stdout/stderr 日志；保留约 24.96 GB 的 BF16/GGUF 正式资产与小型审计 JSON。TD-06 状态保持 `completed`，文件继续留在 `work/task/mvp-model-training/`，不搬入 `work/done/`。

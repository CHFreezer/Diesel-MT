# Hy-MT2 7B teacher 离线运行时

TD-06 已完成。最终冻结腾讯官方 [`tencent/Hy-MT2-7B-GGUF`](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) 的 `HY-MT2-7B-Q8_0.gguf`，通过锁定的 llama.cpp CUDA 后端作为 sequence-level 蒸馏源。质量比较的唯一诊断基线是官方 [`tencent/Hy-MT2-7B`](https://huggingface.co/tencent/Hy-MT2-7B) 原版未量化 BF16 输出，不再使用 FP8 输出作为质量参考。

规范选型见 [`configs/hymt2_teacher_selection.yaml`](../configs/hymt2_teacher_selection.yaml)，三方机器报告见 [`artifacts/model-training/reports/teacher/runtime-comparison.json`](../artifacts/model-training/reports/teacher/runtime-comparison.json)。TD-07 仍须在冻结的人类 dev/reference 上校准 prompt、decode 与逐路由质量；TD-06 的短探针只证明当前诊断输入上的输出一致性。

## 冻结身份与许可证

- 原版 BF16/INT8 来源：`tencent/Hy-MT2-7B` revision `9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223`，15 个运行文件共 16,075,624,007 bytes。
- 最终蒸馏 artifact：`tencent/Hy-MT2-7B-GGUF` revision `ab8472660ac61fac25f1af43fac2599d52a8a775`。
- 文件：`HY-MT2-7B-Q8_0.gguf`，7,981,928,896 bytes，SHA-256 `58b3ad55dd6f6fa08c695cddc34fb5f8f708a844f78ae10508071914b0ed67c0`。
- 后端：llama.cpp release `b10012`、commit `c71854292f7c367cc3b35939f88121d81945472f`、Windows CUDA 13.3、全层 GPU offload、Flash Attention。
- 许可证：两个官方仓库均记录 Apache-2.0；完整文件身份见 [`configs/hymt2_teacher_benchmark.lock.json`](../configs/hymt2_teacher_benchmark.lock.json)。模型许可证不自动解决输入语料、reference、prompt、teacher output 或生成数据的权利边界，TD-08 仍须保留逐样本来源许可与 teacher provenance。

官方原版快照没有 `.py` 或 `auto_map`。官方示例使用 `dtype=torch.bfloat16`、`device_map="auto"`、`trust_remote_code=True`，已原样验证成功；锁定快照最终使用 Transformers 内置 `HunYuanDenseV1ForCausalLM`。离线基准把 `trust_remote_code` 设为 `false`，防止未来浮动远端代码进入运行链。

Chat template 的 SHA-256 为 `788ac16c5d7bfefc28655928ad524c8f378a44cb24d24fb125d6a5859b167677`，没有默认 system prompt。模型卡推荐 `top_p=0.6`，锁定的 `generation_config.json` 为 `top_p=0.8`；TD-07 负责最终 prompt/decode 选择。

## 三方同口径测评

测评运行于 RTX 4060 Ti 16 GB、NVIDIA driver 610.74、Windows 11。协议为 batch=1、context=512、greedy decode、固定 seed、一次 warmup、五标签各 2 次、短探针最多 32 tokens、189 字符容量探针最多 64 tokens，并以 200 ms 间隔采样进程 RSS 与整卡显存。这里的 tokens/s 定义为“生成 token 数 ÷ 端到端单次生成耗时”。

| 指标 | 原版 BF16 | bitsandbytes INT8 | GGUF Q8_0 + CUDA |
| --- | ---: | ---: | ---: |
| 加载时间 | 5.65 s | 12.89 s | 3.28 s |
| 五标签平均延迟 | 2.101 s | 0.995 s | 0.314 s |
| 五标签平均吞吐 | 4.17 tokens/s | 8.79 tokens/s | 27.71 tokens/s |
| 峰值显存增量 | 14,543 MiB | 9,687 MiB | 7,909 MiB |
| 峰值进程 RSS | 17,738,473,472 bytes | 17,030,828,032 bytes | 8,462,090,240 bytes |
| 容量探针吞吐 | 4.53 tokens/s | 9.09 tokens/s | 29.44 tokens/s |
| 五标签输出匹配原版 BF16 | 基线 | 10/10 | 10/10 |

原版 BF16 的 `device_map=auto` 无法在 16 GB 显存中全驻留：`model.layers.29`～`31`、`model.norm` 和 `model.rotary_emb` 被卸载到 CPU。因此其 4.17 tokens/s 是本机可执行的 BF16 + CPU offload 基线，不是全 GPU BF16 速度。INT8 实际加载 `bitsandbytes==0.49.2` 的 `libbitsandbytes_cuda130.dll`，224 个 `Linear8bitLt` 均在 CUDA，无 CPU/disk offload。

GGUF 平均吞吐为原版 BF16 的 6.65 倍、INT8 的 3.15 倍；峰值显存分别少 6,634 MiB 和 1,778 MiB。其运行时与内存优势支持既定的 GGUF 蒸馏源选型。

## 以原版 BF16 为质量基线

五标签短探针中，INT8 与 GGUF 的 10 次输出都和原版 BF16 逐字一致：

| 模型标签 | 原版 BF16 稳定输出 |
| --- | --- |
| `eng_Latn` | `The weather is nice today.` |
| `zho_Hans` | `今天天气很好。` |
| `zho_Hant` | `今天天氣很好。` |
| `jpn_Jpan` | `今日は天気がいいですね。` |
| `kor_Hang` | `오늘 날씨가 좋아요.` |

189 字符日语容量探针中，原版 BF16 与 INT8 逐字一致；GGUF 把其中的 `加熱します` 写为 `調理します`。三者都生成 64 tokens 并触及诊断上限，因此这里只记录量化/后端造成的措辞差异，不声明 GGUF 与 BF16 质量等价，也不把这一个差异判为质量下降。TD-07 已完成 18 路 v1 检查，范围修正后还必须用相同 teacher 与语言名称检查两条新增路线。

FP8 报告 [`artifacts/model-training/reports/teacher/runtime-validation.json`](../artifacts/model-training/reports/teacher/runtime-validation.json) 仅保留为历史运行证据。当前 Windows Transformers/`compressed-tensors` 路径在首次 forward 前把 FP8 模块解压为 BF16，显存接近占满；它既不是硬件 W8A8 FP8，也不是质量基线或正式蒸馏后端。

## 工作目录存储

正式本地 runtime 位于 Git-ignored 的 `artifacts/model-training/runtime/`，不再使用 D 盘旧 staging：

- `teacher/hymt2-7b-gguf-q8/snapshot/`：选定 GGUF Q8_0 与官方仓库元数据。
- `teacher/hymt2-7b-gguf-q8/llama.cpp-b10012-cuda13.3/bin/`：锁定的 llama.cpp CUDA 后端。
- `teacher/hymt2-7b-bf16/snapshot/`：原版 BF16 质量基线；bitsandbytes 从这同一套 BF16 权重加载并动态量化为 INT8，不存第二份模型。
- `teacher/hymt2-7b-bf16/venv/`：BF16/INT8 共用的可迁移隔离 overlay。
- `teacher/hymt2-7b-comparison/reports/` 与 `teacher/hymt2-7b-fp8/reports/`：小型原始 JSON 审计证据。

迁移后保留约 24.96 GB。已删除 FP8 权重、Hugging Face 下载缓存、llama.cpp 下载压缩包、GGUF 测试 venv 和临时日志。HDD 目录只用于模型文件到 RAM/VRAM 的顺序加载和低频只读访问；不要把 checkpoint、随机写缓存或持续日志放入模型目录。

迁移后从该默认路径实际加载通过：原版 BF16 载入 5.40 秒并输出 `The weather is nice today.`；GGUF server 载入 3.52 秒并输出相同译文。

## 复现命令

三条路径必须串行执行，不能同时占用 GPU。原版 BF16 与 INT8 共用同一锁定快照和隔离 venv；INT8 必须在 Python 启动前设置 `BNB_CUDA_VERSION=130`。

```pwsh
$runtime = (Resolve-Path 'artifacts\model-training\runtime').Path
$env:DIESEL_MT_MODEL_RUNTIME = $runtime
$transformersPython = Join-Path $runtime 'teacher\hymt2-7b-bf16\venv\Scripts\python.exe'

# 原版未量化 BF16：性能与质量基线
& $transformersPython scripts\benchmark_hymt2_teacher_variants.py --variant original-bf16

# bitsandbytes LLM.int8
$env:BNB_CUDA_VERSION = '130'
& $transformersPython scripts\benchmark_hymt2_teacher_variants.py --variant bnb-int8

# 官方 GGUF Q8_0 + llama.cpp CUDA
.conda\python.exe scripts\benchmark_hymt2_teacher_variants.py --variant gguf-q8
```

基准配置为 [`configs/hymt2_teacher_benchmark.yaml`](../configs/hymt2_teacher_benchmark.yaml)。原始报告写入 Git-ignored runtime；跟踪的汇总证据记录三份原始报告的 SHA-256。

## 正式边界

- 只验收 batch size 1、context 512 和 M0 不超过 189 字符的受控生成；不作长上下文或并发声明。
- teacher 运行期间不得同时执行 student GPU 训练、TD-14 benchmark 或其他 teacher 进程。
- 选定 GGUF 不执行 Hugging Face remote code，也不得静默回退到 FP8、bitsandbytes、社区量化或其他 teacher。
- TD-06 的确定性探针不能直接升级为 TD-07 的正式 prompt/decode profile。
- 若 TD-07 的冻结 dev 校准发现 GGUF 相对原版 BF16 或人类 reference 的不可接受退化，D0/D1 与任何 addendum 都必须阻塞并形成新的显式选型决策。当前 D0 v1 只保留为真实数据 smoke；D1 v1 接受 39,941/40,032 条并通过 18 路审查、replay 与门禁，但不能单独满足新的 20 路 TD-15/TD-16 输入条件。TD-07 继续使用既有 `Chinese` / `Traditional Chinese` 名称校准新增两条简繁互转路线，不引入 locale-specific prompt。

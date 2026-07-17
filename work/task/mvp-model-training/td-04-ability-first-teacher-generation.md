# task TD-04 schema v4：生成并验收 20 路 ability-first teacher 数据

状态：in progress（正式生成运行中；不得因等待而改配额、prompt 或 source identity）

依赖：TD-03 schema v4、冻结的 TD-06/TD-07 teacher runtime 与 prompt/decode

## 目标

使用 TD-03 的 200,851 条 source bank 和冻结的官方 Hy-MT2 7B GGUF Q8_0，为五标签 20 路生成离散翻译 targets；16 条 source 非 `zho_Hant` 路线各实收 10,000 条，4 条 `zho_Hant -> X` 只按 851 条原生繁体 source 的质量结果实收。允许把 accepted `X -> zho_Hant` pair 反向复用为受控补充，但不得把它记为原生繁体，也不得二次调用 teacher。

## 冻结输入与身份

- 生成合同：`configs/mvp_60m_teacher_generation.yaml`，规范 SHA-256 `3b6f5def5aea5a8b7339f0862d92fcfbe177feab3d38dcdc96ed33c92dc85357`。
- TD-03 紧凑 manifest SHA-256：`c7bff0a6bdc811b0b06358c84ac5accad3922b3746239fa70292810fc7ed4bc4`。
- teacher：官方 `Hy-MT2-7B-GGUF-Q8_0`、llama.cpp `b10012` CUDA、FlashAttention、greedy-v1。
- 跨语言 prompt SHA-256：`9c6dfc6048070fd8846343a49040211fff36ddf74d38992c745ed212ca9dbd8b`。
- 中文转换 prompt SHA-256：`a8edab20edaa3f07d45b940c516066e50d3ec014e3e60b13e80865acfe6f2a86`。
- 正式运行根：`D:\Diesel-MT-Runtime\mvp60-data-v2\td04`；被否决的 v1、旧 M0/D1/TD-16 产物不覆盖。

## 原子边界

本 task 只负责 teacher generation、输出过滤、受控一跳 pair reverse、资源证据和固定人工抽检。不创建 student checkpoint，不读取 FLORES devtest/正式 test，不运行 dev 选择，不调整训练配方。

## 生成与恢复合同

- source 非 Hant 的 16 路各确定性扫描 12,000 个候选，过滤后只取前 10,000 accepted；低于 10,000 即失败，不从别的 source 或低质样本回填。
- source 为 Hant 的 4 路各生成全部 851 个候选，按过滤结果实收，无 minimum/refill。
- append journal 每 256 条 flush+fsync；resume 必须同时匹配 config、TD-03 manifest、source bank 和总 job 数身份。
- 正式 profile 使用 96 parallel slots。实测可将 GPU 提升到约 90% 平均/100% 峰值、约 13 GiB 显存，选择它而不是只提高约 3% 吞吐却进一步压缩显存余量的 128 slots。
- 任一 OOM、损坏 journal、生成身份漂移或异常退出，只有在现有 journal 身份完整匹配时才允许按原命令精确恢复。

## 质量与反向 pair 门

- 逐条运行空输出、prompt echo/解释、source copy、目标脚本、截断、重复、长度比、数字和 placeholder 等冻结过滤。
- accepted `X -> zho_Hant` 可反向为 `zho_Hant -> X`，目标使用原 human source；每条 Hant 出发路线的 reverse 数不得超过该路线原始 Hant teacher accepted 的 50%。
- reverse 继承 semantic group 和 forward job ID，必须保持数字/placeholder，不计原生 Hant，不进行第二次 teacher call。
- 固定人工队列由 `configs/mvp_60m_teacher_review.yaml` 生成：每路 20 条 accepted，加每路最多 20 条 filtered（不足全检）。每条必须有 pass/warning/block 决定；任一 block 阻止 TD-05。

## 产物与完成条件

- `raw-generation.jsonl`、`accepted-teacher.jsonl`、`filtered-teacher.jsonl`、`reverse-pairs.jsonl`、`gpu-samples.jsonl` 与 manifest-last `manifest.json`。
- 人工 `manual-review-queue.jsonl`、`manual-review-decisions.jsonl`、`manual-review-report.json`。
- 20 路均出现；16 条固定路线各 exactly 10,000；四条 Hant 路线为 quality actual；反向比例和 provenance 合法；人工审查无 blocker；所有文件哈希与 generation identity 可追溯。

首次 v1 在 14,246 条处因 KFTT 英文实体臆译被质量门否决并停止。v2 的 256 条 probe 接受 254，抽检 UNPC/ALT/韩英新闻未复现该系统性模式；新 generation identity 为 `e4216e0f5e33a1ff830aa1f0d8f30dd1d76ff2fe8c8e34a335d23f6fba4c0783`。

当前正式生成由 `scripts/generate_mvp_60m_teacher.py generate --runtime-root D:\Diesel-MT-Runtime\mvp60-data-v2` 执行。完成前本 task 保持 `in progress`。

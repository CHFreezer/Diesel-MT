# task TD-04 schema v4：生成并验收 20 路 ability-first teacher 数据

状态：rejected（v3 生成/数量门完成；人工质量门发现 KFTT 日文→英文系统性实体与术语错误）

依赖：TD-03 schema v4、冻结的 TD-06/TD-07 teacher runtime 与 prompt/decode

## 目标

使用 TD-03 的 200,851 条 source bank 和冻结的官方 Hy-MT2 7B GGUF Q8_0，为五标签 20 路生成离散翻译 targets；16 条 source 非 `zho_Hant` 路线各实收 10,000 条，4 条 `zho_Hant -> X` 只按 851 条原生繁体 source 的质量结果实收。允许把 accepted `X -> zho_Hant` pair 反向复用为受控补充，但不得把它记为原生繁体，也不得二次调用 teacher。

## 冻结输入与身份

- 生成合同：`configs/mvp_60m_teacher_generation.yaml`，规范 SHA-256 `d062b22cb853247460051fa131b40d61f5069ce5f93948175db1b1e31a1b847e`。
- TD-03 紧凑 manifest SHA-256：`c7bff0a6bdc811b0b06358c84ac5accad3922b3746239fa70292810fc7ed4bc4`。
- teacher：官方 `Hy-MT2-7B-GGUF-Q8_0`、llama.cpp `b10012` CUDA、FlashAttention、greedy-v1。
- 跨语言 prompt SHA-256：`9c6dfc6048070fd8846343a49040211fff36ddf74d38992c745ed212ca9dbd8b`。
- 中文转换 prompt SHA-256：`a8edab20edaa3f07d45b940c516066e50d3ec014e3e60b13e80865acfe6f2a86`。
- 正式运行根：`D:\Diesel-MT-Runtime\mvp60-data-v3\td04`；被否决的 v1/v2、旧 M0/D1/TD-16 产物不覆盖。

## 原子边界

本 task 只负责 teacher generation、输出过滤、受控一跳 pair reverse、资源证据和固定人工抽检。不创建 student checkpoint，不读取 FLORES devtest/正式 test，不运行 dev 选择，不调整训练配方。

## 生成与恢复合同

- source 非 Hant 的 16 路各确定性扫描 12,000 个候选，过滤后只取前 10,000 accepted；低于 10,000 即失败，不从别的 source 或低质样本回填。
- source 为 Hant 的 4 路各生成全部 851 个候选，按过滤结果实收，无 minimum/refill。
- append journal 每 256 条 flush+fsync；resume 必须同时匹配 config、TD-03 manifest、source bank 和总 job 数身份。
- 正式 profile 使用 64 parallel slots、llama.cpp 总上下文 32,768，即 512 tokens/slot。实测约 13.1 GiB 显存、无请求错误；相同 Hans→KO 长句批次比 96 slots/256 tokens per slot 更快，并把截断从 9/128 降到 4/128。
- 20 路 `max_output_tokens` 由本 TD-04 配置显式锁定，不修改历史 TD-07 prompt。短目标维持 64；风险路线使用 EN 96、JA 112/128、KO 128/192、简繁转换 96，具体逐路值以配置为准。
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

首次 v1 在 14,246 条处因 KFTT 英文实体臆译被质量门否决并停止。v2 修复来源后未复现实体臆译，但在 39,130 条处预判 `eng_Latn→kor_Hang` 只有 75.1% 接收、按 12k scan 无法达到 10k；根因是旧 80-token 输出上限和 256 tokens/slot 上下文造成 24.8% 截断。v2 journal 身份 `e4216e0f5e33a1ff830aa1f0d8f30dd1d76ff2fe8c8e34a335d23f6fba4c0783` 已验证完整并保留，不得恢复或混入 v3。

v3 的隔离预检覆盖剩余 12 条固定路线，并对五条风险路线复测候选上限；64 slots/32,768 总上下文与逐路上限均通过真实 Hy-MT2 请求，数量门不降低、截断输出仍拒收。正式 generation identity 为 `c4d1812f66c9be42a3a16f207444f7ee648e836f39d8867a3851d5999b9535ac`。

v3 正式生成命令为 `scripts/generate_mvp_60m_teacher.py generate --runtime-root D:\Diesel-MT-Runtime\mvp60-data-v3`。生成已完整结束：raw 195,404 条，accepted teacher 163,368 条，16 条固定路线各发布候选 10,000，Hant 原生路线质量实收，reverse 1,683 条；manifest SHA-256 为 `7e5f136e7efde4e98f114c7482183b9e31386f85e8b38441c0a99aed21c10d46`。这只证明运行、长句上限和数量门完成，不代表质量验收。

固定人工队列共 756 条。检查到 200 条时，`jpn_Jpan→eng_Latn` 的 20 条 accepted 中至少 7 条存在实质专名、年号或术语错误，且全部追溯到 `kftt-1.0-en-ja` 日文 source：例如「藤原秀郷／将門」被改成其他人物名、「安永」变成不存在的 `Eiyo`、「チューハイ」变成 `tuhao`。依据 `any-block-prevents-TD05` 合同立即停止，未检查记录没有自动 pass，也没有伪造完整 decisions/report。紧凑阻断证据为 [`mvp-60m-td04-v3-rejected-kftt-japanese.json`](../../../artifacts/model-training/reports/m0/mvp-60m-td04-v3-rejected-kftt-japanese.json)。v3 runtime 保持不可变，禁止 resume、禁止混入后续身份、禁止发布 TD-05。

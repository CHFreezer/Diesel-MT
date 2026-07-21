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

## 独立 DeepSeek 全量审查候选

为评估“逐条看完全部译文”是否能以可接受成本补足 756 条人工队列，新增 `scripts/deepseek_translation_review.py`、冻结配置 `configs/deepseek_translation_review.yaml` 和 30 条独立校准定义。审查只读取 163,368 条 accepted teacher 与 1,683 条 reverse pair，共 165,051 条；不请求重译、不修改 teacher corpus、不读取 FLORES devtest/正式 test。API key 只从进程环境或显式 Git-ignored 本地脚本读取，响应按实际输入文本、prompt、模型和 thinking 身份原子保存，可断点续用。官方接口、JSON 输出与费用口径见 [DeepSeek API 文档](https://api-docs.deepseek.com/) 和 [定价页](https://api-docs.deepseek.com/quick_start/pricing/)。

普通非思考模式首轮只抓到 7/10 已知错例，证明参数量本身不能替代校准。冻结的 `deepseek-translation-fidelity-review-v6-thinking` 在修正一条人工漏标后，对 11 条已知错例抓到 10 条，错误 flag recall 90.91%；19 条已知干净样本全部 pass。唯一已知漏检是把日记专名解释性意译的问题。校准实际使用 13,479 tokens、费用 0.002561 美元；它同时发现 `咫/尺` 被错误写成普通话拼音 `zhi/chi`，该条经复核从 clean 改为真实 `terminology_error`。校准配置、队列、主配置和响应分别绑定 SHA-256，未达 90%/80% 门时脚本拒绝启动全量。

思考模式按 128 条/批冻结，避免 512 条时超过 49,152 completion-token 上限。全量保守估算 1,290 批、22,782,177 input tokens、54,466,830 completion tokens，按 2026-07-18 官方价格为 18.440217 美元；默认 10 美元预检门会阻止误启动。首批 128 条 pilot 实际耗时 158.7 秒、费用 0.008501 美元，得到 123 pass、3 warning、2 reject；5 条经人工复核均为真实问题，包括 `wheel well→エンジンルーム`、遗漏 `possibly`、无来源的“速やか”、`Decade→デー` 和 `Theo→トエ`。据 pilot 外推全量更接近约 11 美元，但仍保留 18.44 美元作为预算上界。

该审查层当前只证明“全量检查可行且初步精度足够”，不撤销 v3 rejected 身份。若用户批准全量费用，应按同一 prompt/config 从已完成 pilot 精确续跑；完成后先人工复核全部 warning/reject 与 pass 分层样本，再将可接受记录发布为新的派生 corpus/manifest 身份。禁止删除原记录后冒充 v3、禁止在审查结果未完成时进入 TD-05。

2026-07-18 阶段化修正：首个 128 条 pilot 仅覆盖 `eng_Latn→jpn_Jpan`，只算 API/JSON/费用工程冒烟，不作为跨路由质量样本。正式阶段顺序改为固定 seed 下按 `route × source_id × teacher/reverse` 分层、桶内稳定 hash 排序、桶间 round-robin；累计扩大时复用所有既有 batch。阶段一累计 512 条，人工全检所有 warning/reject 并分层抽查 64 条 pass；阶段二累计 2,048 条，阶段三累计 8,192 条。每阶段都必须检查严重漏检、误报率、来源/路线聚集、类别分布、completion 上限和实际费用，任一出现系统性漏检、不可接受误报或输出不完整即停止，不自动扩大。只有 8,192 条阶段仍稳定，才向用户申请全量预算。

阶段一已按新顺序覆盖全部20路、60个 `route/source/kind` strata，共512条，费用0.045930美元；DeepSeek给出464 pass、9 warning、39 reject。Codex逐条复核48条flag：43条存在实质错误、来源歧义或合理人工复核价值，5条为过严/可接受变体。继续分层复核64条pass，未发现数字、否定、主体或整句级灾难性遗漏，但发现2条实质漏检和2条边界漏检：台湾法律 `保防工作` 被弱化为普通 `prevention work`，专名 `嵐電` 被错写成 `Arashidenden`，另有 `大家姐` 称谓弱化和 `I.M. Jolly` 转写边界。当前结论为 `hold_before_expansion`：单次Flash对常规错误有效，但对专名/标题转写和地区法律技术术语存在可重复盲点；在设计有界的pass侧二次检查前，不进入2,048条阶段。

## DeepSeek 直译对 Hy-MT2 的 512 条 A/B

为直接回答“继续检查 Hy-MT2，还是改为蒸馏 DeepSeek”这一决策，新增 `scripts/deepseek_translation_ab.py` 和冻结配置 `configs/deepseek_translation_ab.yaml`。A/B 精确复用上述 512 条、20 路、60 strata 样本；DeepSeek 非思考模式只接收 source 和目标语言，不看到 Hy-MT2 译文。两组输出随后各自经过同一思考模式质量审查，并把全部至少一侧被 flag 的 55 对和确定性抽取的 64 对 both-pass 组成 119 条盲评队列。候选身份在全部人工选择完成后才解盲。

结果支持改用 DeepSeek 直译：独立模型审查中 DeepSeek 为 502 pass/2 warning/8 reject，Hy-MT2 为 464 pass/9 warning/39 reject；配对上 45 条改善、7 条退化、3 条两侧均被 flag。119 条人工盲评得到 DeepSeek 胜 56、Hy-MT2 胜 22、双方可接受/平局 33、双方均差或 source 过于破损 8；78 条有明确胜负的记录中 DeepSeek 胜率为 71.79%。该盲评也验证 DeepSeek 更好地保留政党、人地名、极性、组织与细节，但仍发现出闸职务、台湾统计术语、UNDP 名称、目标脚本/语言等 7 个退化案例。

512 条直译实际费用 0.011892 美元，同批独立思考审查 0.044179 美元，总计 0.056071 美元。按本次实际 token 线性外推，195,404 条只做直译约 4.54 美元；若每条再做同等级思考审查，总费用约 21.40 美元。这只是本批 token 分布下的诊断估计，不是供应商报价或预算承诺。

结论从 `hold_before_expansion` 更新为 `deepseek_direct_translation_wins_512_record_ab`：允许按相同 source-only、20 路分层、可恢复身份扩大到 2,048 条复验；尚未授权全量生成，也不允许发布 TD-05。原因是当前盲评只有一名审查者，且 7 个 DeepSeek 退化已证明目标语言检查、术语和实体门仍不可省略。紧凑证据为 [`deepseek-direct-translation-ab-512.json`](../../../artifacts/model-training/reports/m0/deepseek-direct-translation-ab-512.json)；Hy-MT2 v3 rejected 产物保持不变，FLORES devtest/正式 test 均未访问。

## Hy-MT2 官方 sampling 对当前 greedy 的 512 条 A/B

为排除“Hy-MT2 质量差只是因为没有使用官方推荐参数”这一假设，新增 `decode-ab` 隔离入口、`scripts/hymt2_decode_ab.py` 和冻结配置 `configs/hymt2_decode_ab.yaml`。A/B 从 v3 accepted teacher 中确定性抽取 512 条 teacher-only 记录，覆盖20路和9个来源，并强制包含5条已知 KFTT blocker；模型、Q8_0 artifact、llama.cpp/FlashAttention、prompt、source、route、chat template、64 slots、32,768上下文与逐路64～192输出上限均不变，只把 `greedy-v1` 改为官方模型卡推荐的 `temperature=0.7/top_p=0.6/top_k=20/repeat_penalty=1.05`。官方通用 `max_tokens=4096` 没有替换逐路上限，因为512条均正常 stop、零 length finish，放大上限既不能增加有效译文，又会同时改变已验证的并发上下文预算。

512 条中274条译文与 greedy 完全相同，238条发生变化，两侧自动过滤均512/512 accepted。DeepSeek thinking fidelity review 原始得到 greedy 467 pass、sampling 460 pass；相同字符串在两次独立审阅中有13条判定漂移，因此按“相同输出必须同判定”校正后 sampling 为463 pass，仍低于 greedy。只看238条真实变化，7条由 flag 改为 pass、11条由 pass 退化、19条两侧均 flag、201条两侧均 pass。对18条判定变化且译文确实不同的记录再做 source+A/B 盲评，结果为 greedy 8胜、sampling 4胜、5平、1双差。

已知 blocker 的结果更直接：`チューハイ→tuhao` 被修为 `chu-hai`，但 `安永→Eiyo` 与日记专名意译逐字复现；`藤原秀郷/将門` 仍被替换成错误人物名；另一条只改变汉字/长音表面，未完整修复英文专名和年号。因此5条只修复1条，4条仍不合格。sampling 生成期间 GPU 平均/峰值利用率85.063%/99%、峰值功耗129.36W、峰值显存13,917MiB，说明实验确实运行了高负载本地 teacher，而不是空跑；两侧独立审阅总费用0.096627美元。结论保持 `greedy-v1`，不允许用官方 sampling 重跑 v3，也不撤销 TD-04 rejected/TD-05 blocked。紧凑证据为 [`hymt2-decode-ab-512.json`](../../../artifacts/model-training/reports/m0/hymt2-decode-ab-512.json)，正式 test/devtest 均未访问。

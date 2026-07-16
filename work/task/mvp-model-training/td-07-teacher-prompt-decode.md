# task TD-07: 校准 teacher 语言映射、prompt 与解码

状态：completed

依赖：TD-05、TD-06

## 目标

保留 Hy-MT2 7B 的 18 路 v1 校准，继续使用既有 `Chinese` / `Traditional Chinese` 名称，在新冻结 human dev 上为两条简繁互转路线补充唯一、可重放且满足输出格式/脚本门槛的 profile。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 冻结的人类 dev/reference 与 18 路由 manifest
- TD-06 teacher artifact lock、运行 profile 和离线冒烟结果

## 原子边界

本 task 只使用冻结的有界 human dev 子集做校准；不生成正式 train 蒸馏 corpus，所有 dev teacher 输出都必须与 student train 隔离。

## 执行事项

- 固定 `zho_Hans -> Chinese`、`zho_Hant -> Traditional Chinese`、`eng_Latn -> English`、`jpn_Jpan -> Japanese`、`kor_Hang -> Korean` 映射，并分别检查简繁目标脚本。
- 以官方“只输出翻译结果”模板为起点，冻结 prompt version、chat template、system prompt 策略、语言名称和输入分隔方式。
- 在固定 dev/reference 上比较 greedy/确定性解码与官方推荐采样，逐路由报告 chrF/SacreBLEU、脚本合规、额外解释、source copy、空输出和长度比。
- 在查看完整 train 输出前选择唯一规范 profile；不能跨 batch/resume 稳定重放的采样模式不得成为规范配置。
- 保留 18 路 v1 阈值；为两条新增路线冻结最大输入/输出长度、stop 条件，以及允许合法不变文本的路线专用 source-copy 阈值。
- 为 prompt echo、额外解释、错语言/错脚本、繁体退化为简体、截断、重复、占位符损坏和 source copy 建立正反例测试。
- 保存逐样本 raw output/reference 对照及哈希，确保任何校准输出均不进入 student train。

## 产物

- teacher 语言映射和版本化 prompt/decode 配置。
- 不可变 18 路 v1 报告、两路校准 addendum、逐样本对照和输出过滤测试。
- 唯一规范 profile 的可重放验证记录。

## 验收

- 20 路都满足预先声明的质量、脚本和格式门槛；既有 prompt 语言名称不变。
- 同一 artifact、输入和 profile 可按冻结契约重放。
- dev 校准输出与 train corpus 严格隔离，test 从未送入 teacher。
- 任一路由失败都会阻塞 TD-08。

## 实现与验收证据

TD-07 于 2026-07-15 完成。新增 `configs/hymt2_teacher_prompt_decode.yaml`、`scripts/hymt2_distillation.py`、`scripts/calibrate_hymt2_teacher.py` 及对应自动化测试；规范 prompt 使用官方默认英文模板、无 system prompt，并显式区分 `Chinese` 与 `Traditional Chinese`。

从冻结的 33,490 条 human dev 记录中按固定 seed 每路由选择 12 条，共 216 条；greedy 与官方推荐采样各生成一遍，并各自对每路由 2 条、共 36 条做独立重放。最终报告为 `artifacts/model-training/reports/teacher/calibration.json`：

- greedy 宏观 chrF 28.615981、char-SacreBLEU 33.923799、自动接受率 0.995370、脚本合规率 1.0；18 路由无 gate failure。
- 官方采样宏观 chrF 只比 greedy 高 0.014524，低于预先冻结的 +2.0 切换门槛，因此唯一规范 profile 为 `greedy-v1`。
- 两个 profile 的 36 条 replay 均 raw/normalized 逐字一致；test 未读取，全部 calibration output 保持 dev-only。
- 首轮低 chrF 路由的 60 条 source/reference/output 已逐条检查。MASSIVE reference 存在地区实体替换，chrF 只作为诊断下限；3 条非系统性语义风险记录在 `configs/hymt2_teacher_calibration_review.yaml`。
- 繁体脚本检查修正了 OpenCC 将常用字 `吃` 映射为异体 `喫` 的误判：只有出现简体反证且没有任何繁体证据时，才判定繁体目标完全退化为简体。

验证命令：`.conda\python.exe -m pytest tests/test_hymt2_distillation.py -q`，14 项通过；两轮实际校准均从同一锁定 GGUF/llama.cpp runtime 完成，第二轮最终状态为 `complete`。

以上是 18 路 v1 完成记录。

## 两路简繁校准完成记录（2026-07-16）

- 新增 profile 继续使用 `Chinese` / `Traditional Chinese` 与既有 prompt/decode 语义；冻结 dev 每路选 12 条，共 24 条，test 未读取。
- greedy 两路接受率与脚本合规率均为 1.0，chrF 分别为 21.063073/21.289241；官方采样宏观 chrF 只提高 0.227781，低于 +2.0 切换门槛，因此继续选择 `greedy-v1`。
- greedy 与 sampling 各 4 条独立 replay 均 raw/normalized 精确一致；报告 SHA-256 为 `1b9c3fd5b71b56f79ce8233116c58f80479b05f61f8e32d8a139d3c0510dcde8`，v1 校准报告未改写。

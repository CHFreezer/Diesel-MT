# task TD-07: 校准 teacher 语言映射、prompt 与解码

状态：pending

依赖：TD-05、TD-06

## 目标

在冻结的人类 dev 子集上为 Hy-MT2 7B 的 18 个路由选出唯一、可重放且满足输出格式/脚本门槛的语言映射、prompt 和 decode profile。

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
- 为 18 路由分别冻结最大输入/输出长度、stop 条件和异常阈值。
- 为 prompt echo、额外解释、错语言/错脚本、繁体退化为简体、截断、重复、占位符损坏和 source copy 建立正反例测试。
- 保存逐样本 raw output/reference 对照及哈希，确保任何校准输出均不进入 student train。

## 产物

- teacher 语言映射和版本化 prompt/decode 配置。
- 18 路由校准报告、逐样本对照和输出过滤测试。
- 唯一规范 profile 的可重放验证记录。

## 验收

- 18 路由都满足预先声明的质量、脚本和格式门槛。
- 同一 artifact、输入和 profile 可按冻结契约重放。
- dev 校准输出与 train corpus 严格隔离，test 从未送入 teacher。
- 任一路由失败都会阻塞 TD-08。

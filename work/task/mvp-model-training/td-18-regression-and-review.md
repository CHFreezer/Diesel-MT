# task TD-18: 完成统一回归、文档与 review 准备

状态：pending

依赖：TD-01 至 TD-17

## 目标

汇总整个 MVP 训练闭环的身份与证据，在干净环境完成离线回归，更新文档并准备唯一一次 todo 级统一 review。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01～TD-15、TD-16A～TD-16F、TD-17 的实现、artifact、manifest、报告和验证命令
- 当前 README、AGENTS、数据/训练/部署文档

## 原子边界

本 task 只做全链回归、证据收口与 review 准备，不补做未完成的前置 task，不扩大数据/模型范围，也不把准备完成等同于 review 已通过。

## 执行事项

- 整理 human 数据、teacher artifact/生成数据、A/B recipe、student、训练、评测、checkpoint、CT2 和离线包的配置/manifest/hash，生成单一追溯索引。
- 运行完整离线测试和全部慢速集成测试，记录命令、版本、数量、耗时和结果；确认无敏感或大体积运行数据被 Git 跟踪。
- 从干净临时目录验证 fixture 数据构建、teacher 离线 fixture 生成/过滤、M1 短训练/恢复、A/B 校验、评测和 CT2 冒烟。
- 更新 README、AGENTS、数据/训练/部署说明和限制，保持 4 产品语言、5 标签、10 组、20 路、12 个跨语言方向 + 2 个简繁互转操作口径。
- 为 TD-01～TD-15、TD-16A～TD-16F、TD-17 补齐输入、输出、验证命令、产物位置和完成证据，避免相互矛盾的独立报告。
- 确认 tokenizer 根哈希未变化，随机部署 checkpoint 和 M1 过拟合均未被描述为真实翻译质量。
- 准备统一 review 检查表，覆盖 remote code/离线边界、蒸馏 provenance/A-B 公平性、许可/泄漏、恢复、质量、量化和部署风险。

## 产物

- 全链可追溯索引和完整回归记录。
- 更新后的项目文档、限制说明和统一 review 输入。
- 干净环境复现与 Git 边界检查证据。

## 验收

- TD-01～TD-17 均为 completed 且证据完整，不存在跳过项。
- 全部离线与慢速回归通过，工作树不含误跟踪的大体积/敏感产物。
- 文档术语、模型身份、质量边界和蒸馏结论一致。
- 具备对 todo 和完整 task 集合执行一次统一 review 的全部输入；通过 review 后才可归档并标记 done。

# task TD-17: 完成 M3 CTranslate2 回接与量化诊断

状态：pending

依赖：TD-16E

## 目标

将唯一训练后 HF 候选转换为 CTranslate2 float32 与 CPU INT8，验证 49,152 ID 空间、20 路语言控制、量化质量差异和完全离线部署包。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-16E 最终 HF MVP、一次性 test 报告和身份 manifest
- TD-13 固定评测协议
- 已完成的 [CTranslate2 deployment review](../../done/review/ctranslate2-deployment.md) 与现有部署工具

## 原子边界

本 task 只对唯一训练候选做部署兼容和量化诊断，不覆盖随机 checkpoint 的归档记录，不重新选择模型，也不把本机短测宣称为生产性能。

## 执行事项

- 泛化现有 CT2 验证逻辑以接受训练后 HF checkpoint，并创建新的模型训练部署记录。
- 生成 float32 诊断模型和 CPU INT8 验收模型，记录转换命令、版本、耗时、文件大小与 SHA-256。
- 逐 ID 比较 frozen tokenizer、HF embedding/`lm_head`、CT2 float32/INT8 的 49,152 词表及特殊 token。
- 对 20 路执行 tokenize、`target_prefix`、去 prefix、decode 和固定样例推理，拒绝未知目标 token、错脚本、空输出和越界。
- 在查看 INT8 结果前冻结退化容差；按 TD-13 比较 HF/float32/INT8 的 chrF/SacreBLEU、脚本合规和固定样例，逐路由报告差异。
- 记录 CPU 延迟、吞吐、compute type 和模型体积为诊断值，不外推生产性能。
- 生成独立 `tokenizer/` + `model/` 离线包，在新进程、离线标志、socket guard 与 manifest 校验下完成端到端推理。

## 产物

- 训练模型 CT2 float32/CPU INT8 artifact 与转换 manifest。
- 逐 ID、20 路、量化质量和性能诊断报告。
- 自包含离线包和自动化慢速回归。

## 验收

- plan 的 M3 门槛全部满足，49,152 ID 与语言控制保持一致。
- 量化差异在预先冻结容差内；超限则失败并保留诊断。
- 离线新进程在网络阻断下完成 20 路接口回归。
- 随机部署 checkpoint 记录未被覆盖或误称为训练模型。

# task TD-09: 实现编码、collator 与 student 构造

状态：completed

依赖：TD-01；可先使用 schema fixture，完整验收依赖 TD-05

## 目标

实现与冻结 49,152 词表及 M2M100 训练语义一致的 source/target 编码、方向感知 collator 和从零初始化 `mvp_e8_d2_v48k` builder。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01 schema、方向矩阵和模型配置骨架
- [`mvp-tokenizer-v0`](../../../artifacts/tokenizers/mvp-tokenizer-v0/)
- TD-05 的 20 路 fixture/composite manifest（完整验收阶段）

## 原子边界

本 task 只负责单 batch 的数据到模型语义和 student 构造，不实现多步训练循环、checkpoint 或质量评测，也不加载任何第三方模型权重。

## 执行事项

- 仅离线加载冻结 tokenizer，校验根 manifest SHA-256、49,152 稠密 ID、五语言 token 和 fast backend。
- 实现 source 编码与 target labels，固定语言 token/`</s>`、padding mask 和 `-100` loss ignore index 的锁定 Transformers 语义。
- 冻结 source/target 最大长度、截断/丢弃规则，并逐路由累计原始/截断 token 数。
- 实现方向感知 collator，拒绝空文本、同标签、allowlist 外标签、缺失目标 token 和词表越界；两条简繁互转路线必须合法通过。
- 从配置创建 `mvp_e8_d2_v48k`，断言 embedding、`lm_head`、特殊 token、decoder start/generation config 与 tokenizer 一致且权重绑定。
- 固定初始化种子并记录 state-dict 身份；禁止加载微型部署 checkpoint 或第三方权重。
- 使用 20 路 fixture 做 CPU tokenize/collate/forward/backward 冒烟，并测试保存和离线重载。

## 产物

- 模型配置、dataset/encoding/collator 模块与 student builder。
- 初始化身份记录、20 路 fixture 回归和离线重载测试。

## 验收

- 20 路均产生正确 inputs/labels 和有限 loss。
- 模型所有词表相关维度严格为 49,152，特殊 token 与 tokenizer 一致。
- 相同 seed/config 产生可验证的同一初始 state-dict 身份。
- 与 TD-05 一起关闭 M0 的编码前置门槛。

## 实现与运行证据

2026-07-16 完成 TD-09：

- 新增 `scripts/mvp_student.py`，严格校验冻结 tokenizer 的完整文件清单、根 manifest SHA-256、49,152 稠密 ID、fast backend 和五个语言 token；编码固定保留 source/target 语言 token 与 `</s>`，超长序列截断时继续保留该边界，空文本明确拒绝，label padding 固定为 `-100`。
- `DirectionAwareCollator` 对每个 batch 保留 sample/group/route 身份，并逐路由报告 source/target 原始、使用和截断 token；同标签、未知标签、缺失语言 token、词表越界和空 batch 均明确失败，两条简繁互转路线合法通过。
- 从 `configs/mvp_e8_d2_v48k.yaml` 以 seed `20260715` 从零创建正式 M2M100 student；两次独立构造得到相同 state-dict SHA-256 `66897f9c358802b9d39d66e61a8b39fad21236d11744b79df194c26db4da66a3`。输入/输出 embedding 与 `lm_head` 绑定，所有词表维度为 49,152，special token、decoder start 和 generation config 与 tokenizer 一致。
- `scripts/validate_mvp_student.py` 从 TD-05 的正式 20 路 train composite 各取一条固定记录，在 CPU 完成单 batch forward/backward；loss 为 `10.815247535705566`，全部梯度有限。随机 HF checkpoint 保存到 Git-ignored runtime 后完全离线重载，模型 state 与 tokenizer vocab 均未变化。
- 机器可读证据为 `artifacts/model-training/reports/student/encoding-validation.json`；其引用的正式 train SHA-256 为 `ba0a5361fc97ca3ecad76b25ac3cfccfe163be785fb68847d11ae909bddba1d8`，冻结 tokenizer manifest SHA-256 为 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。
- 定向测试：`.conda\python.exe -m pytest tests/test_mvp_student.py tests/test_model_training_contract.py -q`，结果 `30 passed`。

该随机 checkpoint 仅证明编码、梯度和保存/重载链有效，不代表训练或翻译质量。TD-09 与已完成的 TD-05 一起关闭 M0 编码门槛，TD-10 可以开始。

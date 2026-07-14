# task TD-09: 实现编码、collator 与 student 构造

状态：pending

依赖：TD-01；可先使用 schema fixture，完整验收依赖 TD-05

## 目标

实现与冻结 49,152 词表及 M2M100 训练语义一致的 source/target 编码、方向感知 collator 和从零初始化 `mvp_e8_d2_v48k` builder。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-01 schema、方向矩阵和模型配置骨架
- [`mvp-tokenizer-v0`](../../../artifacts/tokenizers/mvp-tokenizer-v0/)
- TD-05 的 18 路由 fixture/manifest（完整验收阶段）

## 原子边界

本 task 只负责单 batch 的数据到模型语义和 student 构造，不实现多步训练循环、checkpoint 或质量评测，也不加载任何第三方模型权重。

## 执行事项

- 仅离线加载冻结 tokenizer，校验根 manifest SHA-256、49,152 稠密 ID、五语言 token 和 fast backend。
- 实现 source 编码与 target labels，固定语言 token/`</s>`、padding mask 和 `-100` loss ignore index 的锁定 Transformers 语义。
- 冻结 source/target 最大长度、截断/丢弃规则，并逐路由累计原始/截断 token 数。
- 实现方向感知 collator，拒绝空文本、同标签、简繁互转、allowlist 外标签、缺失目标 token 和词表越界。
- 从配置创建 `mvp_e8_d2_v48k`，断言 embedding、`lm_head`、特殊 token、decoder start/generation config 与 tokenizer 一致且权重绑定。
- 固定初始化种子并记录 state-dict 身份；禁止加载微型部署 checkpoint 或第三方权重。
- 使用 18 路由 fixture 做 CPU tokenize/collate/forward/backward 冒烟，并测试保存和离线重载。

## 产物

- 模型配置、dataset/encoding/collator 模块与 student builder。
- 初始化身份记录、18 路由 fixture 回归和离线重载测试。

## 验收

- 18 路由均产生正确 inputs/labels 和有限 loss。
- 模型所有词表相关维度严格为 49,152，特殊 token 与 tokenizer 一致。
- 相同 seed/config 产生可验证的同一初始 state-dict 身份。
- 与 TD-05 一起关闭 M0 的编码前置门槛。

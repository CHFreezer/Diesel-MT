# task TD-10: 实现训练循环、采样与运行记录

状态：pending

依赖：TD-09

## 目标

实现可配置、可 dry-run、方向采样可重现且不读取 test 的 student 训练入口，并完整记录优化语义、数据曝光和运行健康状态。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-09 encoding/collator、student builder 和模型配置
- TD-01 的运行目录、配置哈希和 Git 边界

## 原子边界

本 task 只实现训练循环与日志，不实现持久化恢复语义（TD-11）、M1/M2 正式运行（TD-12/TD-16）或独立质量评测（TD-13）。

## 执行事项

- 实现 `scripts/train_mvp_model.py`，支持配置、dry-run、train/dev、固定 seed、设备/精度、梯度累积、gradient checkpointing、梯度裁剪和受控 worker。
- 实现方向感知采样器，记录 batch/step 的路由、epoch、样本位置和实际 token 数；低资源权重只来自冻结配置。
- 固定 optimizer、scheduler、warmup、label smoothing（若启用）、最大 step/token 预算和验证频率，全部进入配置哈希。
- 记录 global/optimizer step、train/dev loss、学习率、梯度范数、吞吐、显存峰值、wall time、截断率和异常跳过数。
- 对 NaN/Inf、OOM、空 batch、数据耗尽、配置/数据哈希变化明确失败，禁止静默跳过并发布候选。
- checkpoint 选择只读取 dev；训练进程和数据加载器不得打开 test split。
- 增加 CPU/小模型单步、梯度累积边界、采样重现、非法 loss 和日志 schema 测试。

## 产物

- `scripts/train_mvp_model.py`、训练核心与方向采样器。
- 结构化 run log/schema 和训练循环自动化测试。

## 验收

- fixture 上可稳定完成多个 optimizer step，loss/gradient 有限。
- 路由曝光、样本顺序、优化参数和运行身份均可从日志重建。
- 异常状态明确失败且不发布候选。
- 自动化测试证明训练代码无法访问 test。

# task TD-10: 实现训练循环、采样与运行记录

状态：completed

依赖：TD-09

## 目标

实现可配置、可 dry-run、方向采样可重现且不读取 test 的 student 训练入口，并完整记录优化语义、数据曝光和运行健康状态。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-09 encoding/collator、student builder 和模型配置
- TD-01 的运行目录、配置哈希和 Git 边界
- `configs/mvp_e8_d2_v48k.yaml` 的设备偏好与可配置资源预算 schema

## 原子边界

本 task 只实现训练循环与日志，不实现持久化恢复语义（TD-11）、M1/M2 正式运行（TD-12/TD-16）或独立质量评测（TD-13）。

## 执行事项

- 实现 `scripts/train_mvp_model.py`，支持配置、dry-run、train/dev、固定 seed、设备/精度、梯度累积、gradient checkpointing、梯度裁剪和受控 worker。
- 运行时探测可用设备、精度和内存，将实际硬件身份写入 run manifest；代码不得按 GPU 型号、固定显存容量或盘符分支。
- 从配置读取设备内存预算/预留/最大利用率、主机与 data loader 内存预算、micro batch、累积、最大长度和 worker；有效设备内存上限取绝对预算、总容量乘最大利用率、总容量减预留三者的最小值，预算缺失或探测容量不足时明确失败。
- 实现方向感知采样器，记录 batch/step 的路由、epoch、样本位置和实际 token 数；低资源权重只来自冻结配置。
- 固定 optimizer、scheduler、warmup、label smoothing（若启用）、最大 step/token 预算和验证频率，全部进入配置哈希。
- 记录 global/optimizer step、train/dev loss、学习率、梯度范数、吞吐、显存峰值、wall time、截断率和异常跳过数。
- 对 NaN/Inf、OOM、空 batch、数据耗尽、配置/数据哈希变化明确失败，禁止静默跳过并发布候选；仅 TD-14 benchmark 模式可按配置的有限 `oom_retry_limit` 搜索候选，正式训练 OOM 立即失败。
- checkpoint 选择只读取 dev；训练进程和数据加载器不得打开 test split。
- 增加 CPU/小模型单步、梯度累积边界、采样重现、非法 loss、资源预算不足、profile 切换和日志 schema 测试，证明调整显存预算不需要修改代码。

## 产物

- `scripts/train_mvp_model.py`、训练核心与方向采样器。
- 结构化 run log/schema 和训练循环自动化测试。

## 验收

- fixture 上可稳定完成多个 optimizer step，loss/gradient 有限。
- 路由曝光、样本顺序、优化参数和运行身份均可从日志重建。
- 异常状态明确失败且不发布候选。
- 自动化测试证明训练代码无法访问 test。
- 自动化测试证明设备容量由配置和运行时探测约束，不包含特定 GPU 型号或固定显存容量常量。

## 实现与运行证据

2026-07-16 完成 TD-10：

- 新增 `scripts/mvp_training.py` 与 `scripts/train_mvp_model.py`，配置 `configs/mvp_training_td10_smoke.yaml` 严格绑定 student/tokenizer/train/dev/manifest 身份、20 路权重、资源预算、batch/累积/长度/worker、AdamW/linear scheduler、step/token 上限以及 validation/checkpoint 频率；训练配置 canonical SHA-256 为 `96392a3019bfbea942c9341724075041716a77765156c4922a62732798fe68e2`。
- dry-run 完成配置、输入哈希、Git 状态、设备/后端/精度和内存容量探测，不构建模型或创建输出。资源验证将绝对设备预算、总容量乘最大利用率、总容量减预留三者取最小值，并独立约束 host/dataloader 内存；CPU/CUDA 选择不含设备型号分支。
- `DeterministicRouteSampler` 使用可恢复的平滑加权轮询和逐路由确定性 shuffle，日志逐 micro/optimizer step 保存 route、epoch、位置、sample/group ID、token、loss、LR、梯度范数、吞吐、内存与截断统计。tokenizer worker 为有界线程池且不预取 sampler，0/2 worker 产生相同有序 batch。
- 训练入口只接受 train/dev 字段，schema 明确拒绝 `test_path`；train/dev 在运行前后校验 SHA-256。空路由、空 batch、数据哈希变化、预算不足、NaN/Inf loss/gradient、token 上限不足和正式训练 OOM 均明确失败；只有 `td14_benchmark` 配置可以声明有限 OOM retry 预算。
- 正式 `mvp_e8_d2_v48k` CPU 冒烟完成 2 optimizer step / 4 micro step / 20 samples / 401 tokens，20 路各曝光一次，无异常跳过；mean/final train loss 为 `10.6890940666199` / `10.5333671569824`，两次 dev loss 为 `10.42934513092041` / `10.289074897766113`，峰值进程常驻内存为 2,179,981,312 B。
- 两次独立正式运行的 loss、step、sampler state 和排除 wall-time/吞吐噪声后的事件语义逐项精确一致；semantic trace SHA-256 为 `b937866624470c1764aacaab155690826eebb0f841d11159d1d83b0ef1236b74`。原始 event 文件哈希可不同，因为其保留真实 wall time/吞吐。
- 机器可读证据为 `artifacts/model-training/reports/student/training-smoke.json`（SHA-256 `3b905c48c9fb66a7155e9e53dff1871f5cd4d72065912d67b6014f2f3bf90d66`）；热日志位于运行 manifest 记录的本机运行根。定向测试 `.conda\python.exe -m pytest tests/test_mvp_training.py tests/test_mvp_student.py tests/test_model_training_contract.py -q` 结果 `37 passed`。

TD-10 不发布候选权重；它只关闭多步训练、采样、资源校验和运行日志语义。TD-11 从该 optimizer/scheduler/sampler 状态接口实现持久化恢复。

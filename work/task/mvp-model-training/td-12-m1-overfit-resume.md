# task TD-12: 完成 M1 小样本过拟合与恢复验收

状态：completed

依赖：TD-05、TD-11

## 目标

使用正式 student 与 20 路微型 fixture 证明模型能学习目标映射、保持语言控制、精确恢复并离线保存/重载，从而关闭 M1。

## 输入

- [MVP model training todo](../../todo/mvp-model-training.md)
- TD-05 冻结的 20 路 fixture/composite manifest
- TD-09～TD-11 的 student、训练和 checkpoint/resume 实现

## 原子边界

本 task 只在微型 fixture 上做过拟合和恢复验收，不把结果解释为真实翻译质量，不调优正式 GPU profile，也不读取正式 test。

## 执行事项

- 用正式 `mvp_e8_d2_v48k` 与固定 fixture 建立随机初始化基线，在看结果前冻结 step/token 预算、解码配置和阈值。
- 在预算内将 mean loss 降至初始基线 10% 以下；每路由至少一条固定记忆样例在固定解码下得到正确目标语言和规范化 exact match。
- 确认 20 路均被采样；路由饿死、错目标语言或空输出视为失败，source-copy 按跨语言/简繁互转各自合同判断。
- 从中途 checkpoint 恢复完成相同预算，与连续运行比较 step、采样、loss、权重和固定生成。
- 保存并离线重载最终 HF checkpoint，验证 tokenizer 未修改、词表仍为 49,152、generation config 完整。
- 记录显存峰值、吞吐、耗时、loss 曲线和固定样例，并标注该结果不代表真实质量。

## 产物

- M1 过拟合 HF checkpoint。
- 连续/恢复对照、20 路生成回归与资源报告。
- M1 验收记录。

## 验收

- plan 的 M1 loss、记忆、路由和恢复门槛全部满足。
- 连续与恢复运行满足 TD-11 的一致性契约。
- 最终 checkpoint 可完全离线重载且 tokenizer 根未变化。
- 未通过前不得启动 TD-14 的正式 GPU profile 冻结。

## 实现与运行证据

2026-07-16 完成 TD-12：

- 在查看训练结果前提交并冻结 `configs/mvp_training_m1.yaml` 与 `configs/mvp_m1_acceptance.yaml`：正式 `mvp_e8_d2_v48k`、CUDA BF16、20 路各一条固定 train fixture、每 step 完整 20 路、300 optimizer step / 250,000 token 上限、step 150 中断、greedy 生成以及 final eval loss/initial eval loss `<= 0.10`、20/20 规范化 exact match、20/20 target language control、零空输出/跨语言 source-copy。
- 初始 student state-dict SHA-256 复验为 `66897f9c358802b9d39d66e61a8b39fad21236d11744b79df194c26db4da66a3`；fixture 初始/final eval loss 为 `10.8114805221558` / `0.157688140869141`，比率 `0.0145852494989926`，显著低于冻结的 0.10 门槛。
- 20 个路由各精确曝光 300 次，共 6,000 samples / 110,400 tokens；20 条固定生成全部得到规范化 exact match 和正确目标语言控制，空输出与跨语言 source-copy 均为 0。两条简繁互转按人类 target exact match 验收，不以跨语言 copy 规则误判。
- 连续 300 step 与 step 150 中断/恢复到 300 的 step、loss、采样器/RNG 和语义 trace 完全相同；最终 model、gradient、规范化 optimizer、scheduler、scaler、RNG、trainer 七个 payload 逐 SHA-256 一致。model/optimizer SHA-256 为 `44a88249b1def5d41b50ab15036c57a7e1af5444dbad68e972f8369034e6feb1` / `49cded4de44fed3f4f9fe9e2e7ec303974efb879813bdff6ac76f2691a50cc00`，trace SHA-256 为 `44ac6ab17d051a1a44f124163afa51b13b6e3200eb657d01b684eba3914fb5f9`。
- 首轮冻结运行曾暴露 CUDA optimizer state 虽逐 tensor 精确相同但 `torch.save` 受 storage 分配影响导致容器哈希不同；未改变训练预算。checkpoint 保存现先按键排序并克隆为连续 CPU tensor，既保留精确 optimizer 语义又使连续/恢复容器逐字节一致；原预算 v2 完整重跑通过。
- M1 HF checkpoint 保存于 run manifest 记录的本机运行根并完全离线重载；state-dict SHA-256 `3cfc2ba0d33afb05f5ec26b4a132f9b491548d58ab55ec13910da36ffabc8273`，M1 manifest SHA-256 `c36da7930ad146de4b0462dd832cdd1d6db97f52082af2edec01990401af42c8`。冻结 tokenizer manifest 未变化，词表仍为 49,152，generation config 与 student 对齐。
- resumed 半程实测 150 step 用时 `71.0140706999955` s，峰值设备内存 `1,218,953,216` B，吞吐 `1,554.62148433081` tokens/s；这些是 M1 小样本事实，不作为 TD-14 profile 常量。
- 机器可读证据 `artifacts/model-training/reports/student/m1-overfit.json` SHA-256 为 `01ba909ad713fbb62c6df90f6d59756bb25da4227215cfc2d7c7a467c15ad6e5`。定向回归 `.conda\python.exe -m pytest tests/test_mvp_m1.py tests/test_mvp_checkpoint.py tests/test_mvp_training.py -q` 为 `18 passed, 1 skipped`；跳过项仍仅为当前 Windows symlink 创建权限条件。

该 checkpoint 只证明 20 条 fixture 可记忆、语言控制和恢复链正确，不代表真实 dev/test 翻译质量。M1 已关闭，TD-14 可以开始资源 profile 基准。

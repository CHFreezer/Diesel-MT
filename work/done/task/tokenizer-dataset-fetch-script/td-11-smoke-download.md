# task TD-11: 执行 smoke 下载

状态：done

依赖：TD-09、TD-10；其余任务通过这两个完成门槛传递依赖。

## 目标

用小字符预算运行真实 HPLT 四语链路，验证网络、缓存、清洗、复现和质量检查闭环。

## 输入

- 已通过测试的下载脚本、配置、source lock 和依赖 lock。
- `smoke` profile。
- 可用磁盘和网络连接。

## 执行事项

- 按环境文档激活 `.conda` 并安装锁定依赖。
- 执行 `smoke --dry-run`，保存配置指纹和操作摘要。
- 执行首次冷缓存四语下载与构建。
- 在独立输出目录使用热缓存和 offline 模式重复构建。
- 比较四语文件和确定性 manifest 的字节及 SHA-256。
- 人工抽查四语样本和质量报告。
- 记录下载量、耗时、峰值磁盘、失败重试和问题。

## 产物

- smoke 四语语料、manifest、run 记录和质量报告。
- 两次构建的哈希对比记录。
- 发现问题和配置调整建议。

## 验收

- 四语真实数据链路完整运行。
- 冷缓存与热缓存/offline 输出字节级一致。
- 每种语言抽样质量可接受，无明显脚本级污染。
- 所有失败和重试均可解释，没有半成品最终文件。

## 验证记录（2026-07-13）

- 首次真实获取已缓存并校验四个 HPLT 锁定前缀（4.25 GiB）；最终 `conservative-v3` smoke 使用同一缓存离线重建。
- 主构建为 16 worker、5.054 s；独立复现为 1 worker、4.326 s。四个 corpus、manifest、report 六项逐字节一致，D: staging 最终均为 0 文件。
- 稳定 SHA-256：eng `97c3db4ca7e4715e73521dc81c9c225291c0c95ceb604c5f2434555c2c704c40`；zho `11c47913b556f47135778752193339768697abee49becaef7ae862a7fe9bd43e`；jpn `9b9882f147ab419fc692164e0df7fadb4e1c211ca2a5bff38b86218b56895eaf`；kor `c744316c1df0fca57f5716afcdb4ffd014bf001212efdaabda375f2f663e2b1c`。
- smoke manifest `b20cdb52a1d678889031a82dc42f10d6872d896e26c629ed9fcb92e90e0f7b22`，报告 `7027e17483565d31966d3580f4179d0f3b4a3cb55c693678593e4ff426fb0b58`；四语人工样本用于确定 v3 规则。

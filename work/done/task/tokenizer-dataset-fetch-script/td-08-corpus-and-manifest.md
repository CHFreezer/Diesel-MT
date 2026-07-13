# task TD-08: 生成训练入口与 manifest

状态：done

启动依赖：TD-04；可提前设计 schema 和原子输出协议。

完成依赖：TD-03、TD-05、TD-07

## 目标

生成可直接训练 tokenizer 的四语文本、确定性 manifest 和独立运行记录。

## 输入

- 四语均衡文本流和处理统计。
- 配置、source lock、代码版本和依赖 lock 指纹。

## 执行事项

- 写入四个目标语料文件，每行一个文本单元，不注入来源标记或语言 token。
- 固定 UTF-8 无 BOM、LF 换行、记录顺序和尾部换行规则。
- 使用临时文件写入并原子替换最终文件。
- 生成字段顺序稳定的 `manifest.jsonl`。
- 记录配置、source lock、Git commit、dirty 状态、相关源码、依赖 lock 和输出文件 SHA-256。
- 将下载时间、耗时、机器路径等易变字段写入独立 `run.json`。
- 避免 manifest 自引用哈希；只记录四语文件及其他非自引用产物哈希。

## 并行边界

TD-05、TD-06 或 TD-07 执行期间，只能并行完成 schema、接口和测试设计。读取真实获取器输出、组装完整流水线和最终验收必须等待 TD-05 与 TD-07 完成。

## 产物

- `data/tokenizer/corpus/mvp/eng_Latn.txt`
- `data/tokenizer/corpus/mvp/zho_Hans.txt`
- `data/tokenizer/corpus/mvp/jpn_Jpan.txt`
- `data/tokenizer/corpus/mvp/kor_Hang.txt`
- `manifest.jsonl` 和 `run.json`

## 验收

- 四语文件非空且编码、换行和文件名符合约定。
- 两次相同构建的语料与 manifest SHA-256 一致。
- 易变运行信息不会改变确定性产物。
- 写入失败不会留下可误判为成功的最终文件。

## 验证记录（2026-07-13）

- 四个最终 `.txt` 已生成；独立全读验证 UTF-8 无 BOM、仅 LF、每行一个单元且文件尾有换行。
- 每语言先在 D: 顺序写入完成 staging，单后台线程按 8 MiB 复制到 E: 临时文件，核对大小/SHA、fsync 后原子替换；所有搬运成功前不发布 manifest。
- `manifest.jsonl` 包含来源/许可证/版本、语言映射、过滤/去重/抽样统计、Git 状态、配置/lock/源码/依赖摘要及输出 SHA；易变路径和时间只在 `run.json`。
- 两个独立 MVP 目录的四语文件与 manifest 字节级一致；manifest SHA-256 `11daef6a8b38c7dc66dc17d51ab4ab2ab9053a0816340261ad76e53740be0477`。

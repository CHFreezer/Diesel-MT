# task TD-04: 实现 CLI 与 dry-run

状态：pending

依赖：TD-01、TD-02、TD-03

## 目标

提供稳定、可诊断的命令行入口，在下载前完整展示计划并校验配置和 source lock。

## 输入

- 配置、registry 和 source lock schema。
- 数据目录和依赖约定。

## 执行事项

- 创建 `scripts/fetch_tokenizer_datasets.py`。
- 支持 `--config`、`--lock`、`--out`、`--profile`、`--dry-run`、`--use-cache`、`--offline` 和 seed 参数。
- 提供显式 source lock 解析或更新命令，和正常构建路径分离。
- 在 `--dry-run` 中输出来源、映射、WDS 范围、字符预算、缓存目录和预期操作。
- 确保 `--dry-run` 不下载数据分片、不创建大文件、不修改 lock。
- 为参数、配置、lock 和路径错误定义稳定的非零退出码及错误信息。

## 产物

- CLI 主脚本。
- 命令帮助文本和最小调用示例。
- dry-run 快照或测试断言。

## 验收

- todo 中约定的 CLI 场景均可解析。
- 相同输入的 dry-run 输出顺序稳定。
- 配置或 lock 无效时在任何下载前失败。
- `--offline` 禁止所有网络访问。

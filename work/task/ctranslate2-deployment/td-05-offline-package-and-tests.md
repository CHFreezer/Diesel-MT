# task TD-05: 离线部署包与自动化测试

状态：pending

依赖：TD-04

## 目标

固化 tokenizer 与 CT2 模型分离的部署布局，在新离线进程中复现 CPU 冒烟，并将关键边界纳入自动化慢速集成测试。

## 输入

- [CTranslate2 deployment todo](../../todo/ctranslate2-deployment.md)
- TD-01 至 TD-04 的已验证产物、报告和日志

## 执行事项

- 定义部署根目录，至少包含独立 `tokenizer/`、`model/`、manifest 和运行说明。
- 从干净临时目录启动新进程，禁用网络并仅使用部署根目录完成 tokenize、translate、去 prefix 和 decode。
- 增加自动化慢速测试，覆盖 checkpoint 生成、转换、逐 ID 词表比较、五语言 prefix 与 int8 CPU 冒烟。
- 区分可提交的生成逻辑/精简日志与不提交的大体积随机权重/转换缓存。
- 汇总版本、命令、哈希、运行时间、已知限制和失败排查方式。

## 验收

- 离线新进程端到端冒烟通过，运行时不访问远端资源。
- 部署 manifest 可验证全部必需文件，缺失或篡改时明确失败。
- 慢速集成测试可重复运行并通过。
- review 记录明确本工作流只验证部署兼容性，不代表翻译质量或生产性能。

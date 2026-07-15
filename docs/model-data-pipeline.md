# MVP 模型数据构建管线

TD-03 将 `configs/mvp_model_data.yaml` 与 `configs/mvp_model_data.lock.json` 锁定的 MASSIVE 1.1 多平行来源转换为统一的 UTF-8/LF 人类平行样本。schema v2 已在独立 `data/model/route20-v2/` 根生成 10 组关系：164,778 条清洗后无向记录、432 条拒绝；TD-03 manifest SHA-256 为 `113a33afa2ca6f73e8e10fbd5a3dab876dd470fbf0e570320edb0961901fe0c7`。第二个完全离线 fresh/resume 构建逐文件字节一致，v1 路径未被覆盖。

## 运行入口

先执行无副作用检查：

```pwsh
.conda\python.exe scripts/prepare_model_data.py --offline --dry-run
```

首次获取锁定归档并构建：

```pwsh
.conda\python.exe scripts/prepare_model_data.py
```

仅使用已校验缓存离线重建，或在失败后复用身份一致的逐 locale checkpoint：

```pwsh
.conda\python.exe scripts/prepare_model_data.py --offline --use-cache
.conda\python.exe scripts/prepare_model_data.py --offline --use-cache --resume
```

`--use-cache` 和 `--offline` 都禁止网络访问。缓存缺失或字节数/SHA-256 不符会明确失败，不会回退到浮动版本或 `latest`。

## 规范输出

- `data/model/corpus/mvp/human_parallel.jsonl`：9 组 v1 规范样本，保留不变。schema v2 产物位于独立 versioned root；每个 MASSIVE `(partition,id)` 对齐组最多生成 10 个无向模型关系样本，反向扩展为 20 个训练路由留给 TD-04。
- `data/model/corpus/mvp/sources/massive-1.1/`：原始 `LICENSE` 与 `NOTICE.md` 字节。
- `data/model/reports/td03-rejections.json`：逐来源、逐标签对的拒绝原因计数。
- `data/model/reports/td03-build.json`：来源/locale/关系计数、20 路由潜在计数、清洗 profile 与输出身份。
- `data/model/corpus/mvp/manifest.json`：完整文件大小、SHA-256 和构建身份；永远最后发布，且 `status=complete` 是唯一完成标记。

运行数据全部被 Git 忽略。只有实现、配置、source lock、fixture、测试和文档进入版本库。

## 身份与对齐

- `sample_group_id` 绑定来源 ID/版本、规范 alignment key 和五个 locale 规范文本的 SHA-256。同一 MASSIVE 对齐关系的 10 个模型关系共享 group ID。
- `sample_id` 进一步绑定 group ID、规范 source/target 标签和文本。
- 身份只使用规范 JSON、稳定来源信息与 SHA-256；不包含绝对路径、执行时间、worker/cache 状态或进程相关值。
- fresh、独立输出目录和 `--resume` 命中差异不会改变规范 corpus、报告或 manifest 字节。

当前 `split` 直接保留 MASSIVE 官方 `train`/`dev`/`test` partition。同 group 的五个 locale 必须 partition 和记录 ID 一致；TD-04 负责冻结 group 级切分策略、去重、反向扩展和跨集合泄漏检查。

## 清洗边界

版本化清洗 profile 位于 `scripts/model_data_pipeline.py`，其 SHA-256 进入 checkpoint 和 manifest 身份。管线只做 NFC、Unicode 空白折叠和保守拒绝：空文本、控制字符、HTML 残留、Unicode replacement character、超长文本、异常重复、明显错误脚本占优及极端长度比。

管线不做小写化、兼容字符折叠、自动简繁转换、假名转换或韩文转写。`zho_Hans`/`zho_Hant` 身份来自锁定且已审计的 `zh-CN`/`zh-TW` 原生 locale；第 10 组直接配对两个人工本地化文本，清洗器不会用字符替换伪造 target。

当前来源只产生 `human_parallel` provenance，并保留 source record 与 alignment key。TD-01 schema 对 `teacher_synthetic` 和 `script_conversion` 使用互斥且更严格的生成 provenance；后续任务不得把这些样本静默写成人类来源。

## 失败与恢复

- 归档支持 `.part` 断点续传，最终文件必须同时匹配锁定字节数和 SHA-256。
- tar 只按 source lock 读取唯一的普通文件成员，不把归档路径解压到文件系统；每个成员再次校验字节数和 SHA-256。
- 逐 locale checkpoint 绑定 config、lock、归档、清洗 profile 和管线版本；损坏或身份不符时重建而不是复用。
- 每个输出先写同目录临时文件并 `fsync`/`os.replace`。发布开始时移除旧完成标记；任何中断都不会留下可被识别为 complete 的新 manifest。

专项验证命令：

```pwsh
.conda\python.exe -m pytest tests/test_model_data_pipeline.py tests/test_model_training_contract.py -q
```

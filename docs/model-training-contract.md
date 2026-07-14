# MVP 模型训练执行契约

本文件说明 TD-01 冻结的 schema、方向、配置身份和存储边界。机器可执行的规范位于 [`scripts/model_training_contract.py`](../scripts/model_training_contract.py)，数据与 student 配置分别位于 [`configs/mvp_model_data.yaml`](../configs/mvp_model_data.yaml) 和 [`configs/mvp_e8_d2_v48k.yaml`](../configs/mvp_e8_d2_v48k.yaml)。若本文与校验器冲突，以通过统一 review 的校验器和配置为准，不允许在下游脚本中维护第二套常量。

## 语言与方向

- 产品语言：Chinese、English、Japanese、Korean，共 4 种。
- 模型标签：`eng_Latn`、`zho_Hans`、`zho_Hant`、`jpn_Jpan`、`kor_Hang`，共 5 个。
- 无向平行语料组：英/日/韩之间 3 组、简体中文与英/日/韩 3 组、繁体中文与英/日/韩 3 组，共 9 组。
- 每组交换 source/target 后形成 18 个有向模型路由；按产品语言汇总为 12 个产品方向。
- `zho_Hans <-> zho_Hant` 是脚本转换，两个方向都被显式拒绝。

任何数据、训练或评测入口都必须调用 `validate_route()` 或消费由同一常量生成的 allowlist。不能把“5 个模型标签”表述成“5 种产品语言”，也不能在中文汇总中丢失简体/繁体明细。

## 规范平行样本

每条规范样本必须且只能包含以下基础字段：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 由稳定来源身份和规范内容生成的样本身份 |
| `sample_group_id` | 在方向扩展和 split 前绑定同一对齐/派生关系的 group 身份 |
| `source_id` / `source_version` | source registry 与 lock 中的来源身份 |
| `license` | 该样本继承的许可标识 |
| `src_lang` / `tgt_lang` | 5 标签 allowlist 中的有效跨语言路由 |
| `source_text` / `target_text` | 非空 UTF-8 文本 |
| `split` | `train`、`dev` 或 `test` |

唯一可选顶层字段是 `provenance`。人工平行、teacher synthetic 和脚本转换使用不同的严格字段集：

- `human_parallel`：`kind`、`source_record_id`、`alignment_key`；
- `teacher_synthetic`：另行记录 teacher model/revision、prompt version、decode config hash 和 generation manifest hash；
- `script_conversion`：另行记录转换工具/version、上游 sample ID 和 generation manifest hash。

未知字段、缺失字段、空文本、非法 split、同标签路由、简繁互转和 allowlist 外标签都必须 fail-fast。下游不得通过忽略字段来兼容浮动 schema。

## 配置身份

两个 YAML 文件在解析后都使用同一规范表示：UTF-8 JSON、key 排序、紧凑分隔符、结尾一个 LF；SHA-256 覆盖整个解析后的配置。YAML 注释和键的书写顺序不进入身份，任何语义字段变化都会改变配置哈希。

`mvp_model_data.lock.json` 必须绑定完整数据配置哈希，并锁定启用来源的顺序、下载 URI、版本、许可、归档大小/SHA-256、选中文件大小/SHA-256 和实测对齐统计。正式构建不得解析 `latest`、浮动分支或未进入 lock 的文件。

`mvp_e8_d2_v48k` 固定从零初始化，词表为 49,152，`d_model=512`、encoder 8 层、decoder 2 层、FFN 2,048、8 heads，tokenizer manifest SHA-256 为 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。micro batch、累积、gradient checkpointing、最大 source/target 长度、worker、optimizer/scheduler 和正式预算必须等到 TD-14 本机基准后才能冻结。

## 存储与 Git 边界

| 类型 | 规范位置 | Git 策略 |
| --- | --- | --- |
| 原始下载 | `data/model/raw/` | 忽略，仅保留 `.gitkeep` |
| 可复用缓存 | `data/model/cache/` | 忽略 |
| 中间状态 | `data/model/interim/` | 忽略 |
| MVP corpus | `data/model/corpus/mvp/` | 忽略 |
| 本机详细报告 | `data/model/reports/` | 忽略；精简结论写入 `docs/`/task |
| 默认热运行根 | `artifacts/model-training/runtime/` | 忽略 |
| HF/CT2 发布权重 | `artifacts/models/mvp_e8_d2_v48k/` | 忽略 |
| schema/config/lock/fixture | `scripts/`、`configs/`、`tests/fixtures/` | 可提交 |

默认热运行根是仓库相对路径；本机正式训练可通过 `DIESEL_MT_MODEL_RUNTIME` 指向 D: NVMe 的绝对目录。这个 override 只改变物理 I/O 位置，不改变语义配置哈希；每次 run manifest 必须记录环境变量名、解析后的绝对路径、Git commit/dirty 状态和配置哈希。publish 路径是固定逻辑位置，只有全部文件完成校验后才能从 staging 原子发布。

任何仓库内配置路径必须使用规范 POSIX 相对路径，禁止盘符、反斜杠、`..`、绝对路径或逃逸固定根目录。外部 SSD 路径只允许通过命名环境变量提供，并在运行记录中显式解析。

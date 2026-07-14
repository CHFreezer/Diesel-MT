# task index: CTranslate2 deployment validation

状态：done

## 来源

- plan：[CTranslate2 deployment validation](../../../plan/ctranslate2-deployment.md)
- todo：[CTranslate2 deployment validation](../../todo/ctranslate2-deployment.md)
- 合并机器记录：[deployment-validation.json](../../../../artifacts/ctranslate2/deployment-validation.json)
- 延期来源：[tokenizer TD-10](../mvp-tokenizer/td-10-ctranslate2-smoke.md)

## 依赖图

```mermaid
flowchart LR
    TD01["TD-01 HF checkpoint"] --> TD02["TD-02 CT2 conversion"]
    TD02 --> TD03["TD-03 vocab integrity"]
    TD03 --> TD04["TD-04 CPU inference"]
    TD04 --> TD05["TD-05 offline package/tests"]
```

## 执行顺序

| 阶段 | 编号 | 子任务 | 完成门槛 | 状态 |
| ---: | --- | --- | --- | --- |
| 1 | TD-01 | [微型 HF checkpoint](td-01-hf-checkpoint.md) | checkpoint 可重复生成、离线重载，模型维度匹配 | done |
| 2 | TD-02 | [CTranslate2 转换](td-02-model-conversion.md) | float32 与 CPU int8 转换/加载成功 | done |
| 3 | TD-03 | [词表与 ID 空间完整性](td-03-vocab-integrity.md) | tokenizer/HF/CT2 逐 ID 与特殊 token 一致 | done |
| 4 | TD-04 | [五语言 CPU 推理冒烟](td-04-cpu-inference-smoke.md) | 五目标语言 prefix、推理、decode 通过 | done |
| 5 | TD-05 | [离线部署包与自动化测试](td-05-offline-package-and-tests.md) | 新离线进程和慢速测试通过，文档齐全 | done |

## 状态约定

- `pending`：尚未开始或依赖未完成。
- `in_progress`：正在执行并记录负责文件/产物。
- `completed`：单个 task 的实现和验收已完成，可供后续 task 使用，但尚未进入 review。
- `review`：仅在 TD-01 至 TD-05 全部 completed 后，对整个 todo 和完整 task 集合统一复核。
- `done`：整个 todo 的统一 review 通过，todo、task 和 review 一并归档。

该工作流为严格串行链路；后续任务消费前一任务产生的本地 checkpoint、转换产物或校验报告，不并行修改同一运行目录。单个 task 完成后直接进入 `completed`，不得提前创建独立 review。

TD-01 至 TD-05 已通过 todo 级统一 review，并随 todo 整体归档。

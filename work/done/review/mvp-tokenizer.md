# review: mvp tokenizer

状态：approved / archived

评审日期：2026-07-14

## 评审对象

- plan：[mvp tokenizer](../../plan/mvp-tokenizer.md)
- todo：[mvp tokenizer](../todo/mvp-tokenizer.md)
- task 索引：[mvp tokenizer tasks](../task/mvp-tokenizer/index.md)
- 冻结验收：[mvp-tokenizer-v0 freeze acceptance](../../../artifacts/tokenizers/reports/mvp-tokenizer-v0/freeze_acceptance.md)

本次 review 只评审最终确认的有界范围：五语 49,152 词表 tokenizer 的一次性重训、评测、集成验证和冻结。原路线中的 CTranslate2 转换、CPU 推理及部署打包已移到下游模型/部署工作流，不属于本次 done 门槛。

## 复核结果

- 冻结脚本重新校验成功，状态为 `frozen`；artifact manifest SHA-256 为 `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`。
- `tokenizer.json` SHA-256 为 `22bceccab939afe1003d1fbdd95d9d4e78eb954b2e9003d70131155666b1201c`，最终词表为 49,152，ID 空间与微型 M2M100 embedding/lm_head 一致。
- 独立 holdout 覆盖五种语言；保存/重载词表与 backend 相等，五个 language token 均通过验证，五个微型 M2M100 forward loss 均为有限值。
- 执行 `.\.conda\python.exe -m pytest -q`，结果为 `45 passed in 31.21s`。
- 评审开始前工作区无未提交修改；复核生成的时间字段未作为冻结根身份，冻结根仍由不可变的 `artifact_manifest.json` 决定。

## 非阻断项

- `byte_fallback=false` 时，must-cover alphabet 外的罕见 Unicode 仍可能进入 `<unk>`；冻结报告已通过 offset mapping 记录字符损失。
- 繁体中文 holdout 的 source loss 为 `0.043771%`、roundtrip 为 `95.6%`，已作为当前数据和词表规模下的已知限制接受。
- TD-10 CTranslate2 仍为 `deferred`；已由新的 [CTranslate2 deployment validation](../../plan/ctranslate2-deployment.md) 工作流承接，完成前不得把它描述为已通过的部署验收。

## 结论

未发现阻断 tokenizer 冻结的问题。`mvp-tokenizer-v0` 满足调整后的 plan、todo 和 TD-01 至 TD-09/TD-11 验收边界，批准将 todo、task 和本 review 记录归档到 `work/done/`。后续 tokenizer 调整必须生成新版本和新的冻结记录，不能原地修改 `mvp-tokenizer-v0`。

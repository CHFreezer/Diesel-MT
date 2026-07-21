# task TD-03: 构建近期 human parallel 预审语料

状态：pending

依赖：TD-02B `proceed`

## 目标

按 TD-02B 冻结的来源、版本、扫描上限和预算，全量或分 tranche 构建 human parallel 预审语料。该阶段只做可确定、廉价、可复现的处理，把语义质量判断留给 TD-04。

## 原子边界

不调用 DeepSeek、不修改译文、不生成 teacher target、不发布可供训练消费的最终 corpus、不访问 FLORES devtest/正式 test。旧 schema v4 source bank 与 TD-04 v1/v2/v3 运行根不得覆盖或混入。

## 实现原则

- 复用 `artifact_io.py`、现有数据合同和统一 CLI；来源适配器进入共享数据模块，不为每个数据源新增一个一次性 CLI。
- 下载支持断点续传、大小/hash/成员校验和 manifest-last；浮动上游必须先解析为稳定版本或 commit。
- 规范样本保留 `sample_id/group_id`、两侧稳定 `text_id`、document/work identity、source/version/license、src/tgt 标签、原文、译文、snapshot date、可空 content date 与其证据状态、domain、原始文件/行号、处理版本，以及逐侧 `tokenizer_candidate_status/reason`。内容年代未知的样本可进入普通候选，但不能计入近期内容层。
- 清洗不做英文小写、简繁转换、假名转换、韩文转写或语义改写；粤语不映射为 `zho_Hant`。
- 先按 semantic group 形成组件，再切分 train/dev/test；正反方向、exact/near duplicate 和同一文档/作品不得跨 split。
- 硬门逐项记录拒绝原因；超长数据只能拒绝或进入独立长句 tranche，不能静默截断。
- 预审输出按来源/路线/年份/领域分片，便于 TD-04 构建同质长上下文批次，避免在一个 prompt 中混杂过多任务定义。
- 原始真人两侧与有向训练展开分离保存：同一文本无论参与多少关系或正反路由都只有一个稳定单语身份。未来 64k 工作只能从最终 train 分区的合格真人文本重新抽取并去重，不能消费 dev/test、tokenizer holdout、synthetic、canary、quarantine 或 DeepSeek 改写文本。

## 产物

- 不可变 raw/cache 身份与下载 manifest；
- 规范化 `preaudit.jsonl` 分片和 group/split 索引；
- 硬门拒绝集、统计报告与 TD-04 batch inventory；
- 只含稳定文本引用、标签、来源、许可、日期、领域、split 和资格理由的 tokenizer 候选账本；它不构成 64k tokenizer 训练集或完成产物；
- 不含 API key 或大体积正文的紧凑提交内证据。

## 完成条件

- 所有文件和样本均可追溯到 TD-02A/TD-02B 锁定来源；未知版本或许可按合同 fail closed，未知内容年代必须显式标记并从近期内容统计排除。
- 两次独立构建的规范预审数据、split、统计和 manifest 字节一致。
- 零截断、测试隔离、简繁/粤语和跨 split 去重门全部通过。
- tokenizer 候选账本能追溯到未截断真人原文，且 directed reverse expansion 不会放大单语计数；所有评测/holdout 文本均明确不可用于未来 trainer。
- 预审语料不能被训练 CLI 当作正式 corpus；只有 TD-05 发布身份可解锁训练。

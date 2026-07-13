# task TD-02: 建立目录与 Git 边界

状态：done

依赖：TD-01

## 目标

建立数据运行目录、测试 fixture 和依赖锁定方式，确保大体积语料与本机环境不会进入 Git。

## 输入

- TD-01 的配置结构。
- [Python 环境约定](../../../../docs/python-environment.md)。
- 当前 `.gitignore`。

## 执行事项

- 建立 `data/tokenizer/raw/`、`cache/`、`interim/`、`corpus/mvp/` 和 `reports/` 目录语义。
- 为运行时数据添加 `.gitignore` 规则，只保留必要的目录说明或小型 fixture。
- 建立 `tests/fixtures/tokenizer_datasets/`，准备后续四语无网络测试入口。
- 选择项目依赖声明和 lock 方案，固定直接及传递依赖版本。
- 记录依赖 lock 的 SHA-256，并保证环境可从声明文件重新创建。
- 使用代表性临时文件验证 Git 忽略规则。

## 产物

- 数据目录边界和 `.gitignore` 规则。
- 小型 fixture 目录。
- 依赖声明及依赖 lock。

## 验收

- `git status` 不显示 `.conda`、下载分片、缓存、中间文件或最终语料。
- fixture 和依赖声明可以正常进入 Git。
- 全新 `.conda` 环境可依据声明安装依赖。
- 依赖 lock 的校验值有稳定记录。

## 验证记录（2026-07-13）

- 已建立 `data/tokenizer/{raw,cache,interim,corpus/mvp,reports}` 运行目录及 `tests/fixtures/tokenizer_datasets/` 四语 fixture。
- `.gitignore` 覆盖 `.conda`、Python 测试缓存、`data/tokenizer/` 各运行目录和 `data/tokenizer-*/` 独立复现目录；`git ls-files data` 为空，代表性文件均由 `git check-ignore -v` 命中。
- 依赖由 `requirements.txt` 声明、`requirements.lock` 锁定；lock SHA-256 为 `c00869a790d36c1d147612e4e0b05e21364ef1e2bec4d42ac37b317d79928565`。
- `.conda\python.exe -m pip check` 和锁文件 dry-run 安装检查通过；`git status --short` 不枚举任何大数据文件。

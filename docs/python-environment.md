# Python 环境约定

## 适用范围

本文档定义 Diesel-MT 本地开发、脚本执行和后续依赖管理使用的 Python 环境约定。除非任务文档另有说明，仓库中的 Python 命令均应在该环境中执行。

## 固定约定

- 使用 Conda 管理 Python 环境。
- 使用 PowerShell 7.6 加载 Conda hook 和执行项目命令；其启动程序为 `pwsh.exe`。
- Python 主次版本固定为 `3.11`；允许在 `3.11.x` 范围内升级补丁版本。
- 环境使用项目内 prefix，固定目录为仓库根目录下的 `.conda`。
- `.conda/` 是本机生成内容，必须由 `.gitignore` 排除，不能提交到 Git。
- 安装 Python 包时使用 `python -m pip`，避免调用到基础环境或其他环境中的 `pip`。
- 不依赖 Conda `base` 环境中的第三方包，也不把项目依赖安装到 `base` 环境。

当前已验证的环境：

```text
Python 3.11.15
pip 26.1.2
environment: <repository>/.conda
```

## 创建环境

在仓库根目录通过 `pwsh.exe` 启动 PowerShell 7.6，加载当前工作站的 Conda hook，然后按本地路径创建环境：

```pwsh
& 'C:\Users\chfre\miniconda3\shell\condabin\conda-hook.ps1'
conda create --prefix (Join-Path $PWD '.conda') python=3.11 pip -y
```

如果 `.conda` 已经存在，不应重复创建；直接按下一节激活。

## 激活环境

每个新的 PowerShell 7.6 会话都需要执行：

```pwsh
& 'C:\Users\chfre\miniconda3\shell\condabin\conda-hook.ps1'
conda activate (Join-Path $PWD '.conda')
```

执行前应确认当前目录为仓库根目录。路径包含空格，因此脚本和配置中不得通过未加引号的字符串拼接环境路径。

## 验证环境

激活后执行：

```pwsh
python --version
(Get-Command python).Source
python -m pip --version
```

解释器路径必须指向：

```text
<repository>\.conda\python.exe
```

如果解释器指向 Miniconda `base`、系统 Python 或其他项目目录，应停止安装和运行操作，重新加载 hook 并激活本项目环境。

## 依赖约定

- 所有安装和运行命令必须在已激活的 `.conda` 环境中执行。
- 临时验证依赖也应安装到 `.conda`，不能安装到全局环境。
- 新增项目必需依赖时，必须同步更新仓库中的依赖声明文件；在正式引入依赖管理文件后，以该文件作为可复现安装的唯一依据。
- 不把 `.conda` 目录复制到其他机器。跨机器复现时重新创建环境并安装已声明依赖。
- 使用 `python -m pip install ...`，不使用裸 `pip install ...`。

## 已知限制

当前仓库绝对路径包含空格，Conda 创建环境时会给出路径警告，但 Python 和 pip 已验证可正常运行。后续若某个原生编译工具或第三方脚本不能处理空格路径，应记录具体失败工具和错误，再决定是否为该工具使用独立的无空格环境路径；不能在没有验证依据时改变项目默认环境约定。

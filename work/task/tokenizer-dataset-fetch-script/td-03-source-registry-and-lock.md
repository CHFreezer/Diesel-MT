# task TD-03: 建立数据源 registry

状态：pending

依赖：TD-01

## 目标

建立可校验的数据源 registry 与不可隐式更新的 source lock，固定构建实际消费的远端输入。

## 输入

- TD-01 的数据源与 profile 配置。
- HPLT 3.0 官方 map 文件。
- plan 中的稳定复现契约。

## 执行事项

- 定义 registry 字段、类型、必填规则和语言映射规则。
- 登记四个默认启用的 HPLT 3.0 来源。
- 登记默认禁用的 Wikimedia、FineWeb 和 FineWeb2 备用来源。
- 对许可证、版本、下载入口或语言映射缺失执行快速失败。
- 定义 source lock schema，固定 map SHA-256、分片 URL、逻辑顺序、文件大小和分片 SHA-256。
- 规定显式解析和更新 lock 的流程；正常构建只读 lock。
- 确保 lock 不含凭据、查询令牌、本机路径和运行时间。

## 产物

- 可由脚本读取和验证的数据源 registry。
- `configs/tokenizer_datasets_mvp.lock.json`。
- registry 与 lock schema 测试样例。

## 验收

- 相同远端 map 解析得到字节级一致的 canonical lock。
- lock 中分片顺序明确，字段排序和 JSON 编码规则固定。
- 任一关键字段缺失或校验变化会被拒绝。
- 正常构建不会访问或更新远端 map。

# task TD-12: 执行 MVP 下载

状态：done

依赖：TD-11

## 目标

在 smoke 验收通过后生成正式 MVP tokenizer 四语训练语料，并完成复现和审计验收。

## 输入

- smoke 验收记录和已批准的配置调整。
- 最终 `mvp` profile、source lock、代码版本和依赖 lock。
- 经确认的磁盘、网络和运行时间预算。

## 执行事项

- 冻结最终配置和所有指纹，运行正式 dry-run。
- 核对预计下载量、剩余磁盘和缓存位置。
- 执行完整 MVP 下载、清洗、去重、均衡抽样和报告生成。
- 复核来源版本、许可证、语言映射、字符预算和质量统计。
- 在独立输出目录完成第二次构建和逐文件哈希对比。
- 人工抽查四种语言，并记录接受或回退决定。
- 将最终命令、配置版本、哈希和结果摘要写入报告。

## 产物

- `data/tokenizer/corpus/mvp/` 正式四语训练入口。
- 确定性 manifest、独立运行记录和质量报告。
- 字节级复现验证记录。

## 验收

- 四语语料满足字符预算、语言质量和去重要求。
- source lock、配置、代码和依赖指纹齐全可追溯。
- 两次独立构建的四语文件与 manifest SHA-256 一致。
- 大体积数据未进入 Git，可直接进入 tokenizer 训练 task。

## 验证记录（2026-07-13）

- 第一次：16 worker，D: `Diesel-MT-tokenizer-stage`，耗时 418.118 s，峰值主进程 RSS 8,348,467,200 B（7.78 GiB），最低系统可用内存 86,927,015,936 B；第二次：独立输出目录、8 worker、独立 D: staging，耗时 661.518 s。
- 两次都使用 48 GiB RSS 上限、32 GiB 最低可用内存、`--use-cache --offline`；四个 D: staging 最终均为 0 文件。最终语料合计 9,136,480,537 B（8.51 GiB）。
- 两次逐文件全读 SHA-256/大小相同：
  - eng：1,823,141 行，1,000,000,000 字符，`a9a4eb1a066f6029e7b3d1e1234f0962e932a9c893357f101ca442af085cd47d`
  - zho：3,265,237 行，999,999,995 字符，`f86d1e6fc7d4336ec0c104580270b31bd0bd2254dacbdcbf4a673cdc87ba76c9`
  - jpn：4,202,923 行，999,999,997 字符，`d0bc81514969b14365fa2b1b38de3163f6824be397db6efcbd49e29e522fd751`
  - kor：2,241,401 行，999,999,992 字符，`884f68cba97c7ce06b5616886df27a1a0f3cb6cee2c83e0523018eacec8b30a3`
- manifest `11daef6a8b38c7dc66dc17d51ab4ab2ab9053a0816340261ad76e53740be0477`，报告 `feb8aa7546755dfb6f87fc0ed6e0a4a620c625872eacca2cbecf5953df8d783b`；80 条固定样本按 TD-09 结论接受，可进入 tokenizer 训练。
- 正式命令使用 `.conda\python.exe scripts\fetch_tokenizer_datasets.py --config configs\tokenizer_datasets_mvp.yaml --lock configs\tokenizer_datasets_mvp.lock.json --out data\tokenizer --cache-dir data\tokenizer\cache --profile mvp --concurrency 16 --staging-dir D:\Diesel-MT-tokenizer-stage --max-memory-gib 48 --min-available-memory-gib 32 --use-cache --offline`；复现命令只将 `--out` 改为 `data\tokenizer-mvp-repro`、`--concurrency` 改为 `8`、staging 改为独立目录。

# Bundled Market Snapshots / 内置测试行情快照

This directory holds the read-only market fixtures used for offline examples and integration validation (18 tracked files, 55,413,122 bytes / 52.85 MiB). Files are stored directly in Git—no Git LFS or live network download is required.

本目录存放用于离线示例与集成验收的只读测试行情快照（共 18 个跟踪文件，55,413,122 字节 / 52.85 MiB）。文件直接保存在普通 Git 中，加载时无需 Git LFS 或联网拉取行情。

| Directory / 目录 | Scope / 覆盖范围 | Included artifacts / 包含文件 |
| --- | --- | --- |
| `us_equity_daily` | 500 US equities (`2025-07-01` – `2026-06-30` study window) / 500 只美股 | `panel.parquet`, `membership.parquet`, `manifest.json` |
| `a_share_pit` | 365 China A-shares with raw signal & `hfq` execution prices / 365 只 A 股（原始信号价与后复权执行价） | `panel.parquet`, `membership.parquet`, `manifest.json` |
| `crypto_spot_daily` | 73 crypto spot pairs (`2015` – `2025` UTC daily history) / 73 个加密现货交易对 | `panel.parquet`, `manifest.json` |
| `crypto_perpetual` | `BTCUSDT`, `ETHUSDT`, `SOLUSDT` hourly archive (`2025`) / 3 合约 2025 小时归档 | `trade.parquet`, `mark.parquet`, `funding.parquet`, `manifest.json` |
| `crypto_perpetual_scale` | 64 USDT linear perpetual contracts (`2025` hourly archive) / 64 合约 2025 小时归档 | Same three Parquet tables, `manifest.json`, and 2 funding-transition JSON proofs |

Verify snapshot integrity from the repository root on Linux / 在 Linux 仓库根目录核验文件校验和：

```bash
sha256sum --strict --check data/SHA256SUMS
```

Market presets in `foundry.open(...)` default to these directories (`crypto_perpetual` defaults to the 3-contract fixture; pass `data_dir="./data/crypto_perpetual_scale"` for the 64-contract fixture). To prepare your own datasets, see [Data contracts](../guides/data.md) ([中文数据契约指南](../guides/data.zh-CN.md)).

`foundry.open(...)` 的各市场默认路径指向上述目录（`crypto_perpetual` 默认加载 3 合约快照；传入 `data_dir="./data/crypto_perpetual_scale"` 可加载 64 合约快照）。接入自有数据的格式规范与转换示例见[数据契约指南](../guides/data.zh-CN.md)（[English](../guides/data.md)）。

## Provenance and limitations / 数据来源与局限披露

### English
- **US equities:** legacy vendor snapshot; the manifest does not certify the upstream provider, original announcement timestamps, or revision vintages.
- **China A-shares:** derived from Tencent daily raw and cumulative back-adjusted (`hfq`) series; historical intraday announcement timestamps and revision vintages are unavailable and disclosed as such.
- **Crypto spot:** extracted from the preserved Quantiacs daily archive (source path and digest recorded in `manifest.json`).
- **Crypto perpetuals:** imported from official Binance USD-M public archives with verified checksums and funding-interval transition proofs; raw acquisition `.zip` archives are not duplicated here.
- **Scope:** these fixtures exist for reproducible execution and numerical validation, not as survivorship-free vendor PIT archives or evidence of out-of-sample profitability. The repository's MIT license applies to code and documentation, not to upstream market-data terms.

### 中文
- **美股：**既有历史供应商快照，清单不背书原始供应商身份、公告发布时间戳或历史修订版本。
- **A 股：**源自腾讯日线原始价与累计后复权（`hfq`）序列，缺少历史公告精确发布时间与财务重述版本，已在 `DataContract` 中如实披露。
- **加密现货：**提取自本地保留的 Quantiacs 日线面板，`manifest.json` 记录了源归档路径与哈希。
- **加密永续：**由 Binance 官方公开归档经校验和核验导入，附带资金费结算周期变更证据；原始下载 `.zip` 包不重复纳入仓库。
- **适用边界：**上述数据仅供可复现回测与数值一致性验证，不构成完整商业级 PIT 数据库或样本外盈利证明；仓库 MIT 许可证不替代上游行情数据条款。

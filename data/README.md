# Test market snapshots / 测试行情快照

These are the market inputs used by the Colab PRO 6000 integration validation. The 18 files total 55,413,122 bytes (52.85 MiB); ordinary Git contains the actual files, with no Git LFS download or live market request required to load them. Source manifests and the two funding-transition proofs are included.

这些快照用于 Colab PRO 6000 集成验证。18 个文件合计 55,413,122 字节（52.85 MiB），通过普通 Git 直接保存；加载时无需 Git LFS 或在线行情接口。包含来源清单及两份资金费周期变更证据。

| Directory / 目录 | Coverage / 用途 | Files / 文件 |
| --- | --- | --- |
| `us_equity_daily` | US equities, 500 assets in the configured study / 配置研究中的 500 个美股资产 | Daily panel, historical membership, manifest / 日线、历史资格、清单 |
| `a_share_pit` | A-shares, 365 assets / 365 个 A 股资产 | Raw signal prices, cumulative adjusted execution prices, membership, manifest / 原始信号价、累计复权执行价、资格、清单 |
| `crypto_spot_daily` | 73 spot assets, history from 2015 through 2025 / 73 个现货资产、2015 至 2025 历史 | Preserved daily OHLCV and eligibility, manifest / 保留的日线及资格、清单 |
| `crypto_perpetual` | BTC/ETH/SOL, 2025 hourly history / 3 合约、2025 小时历史 | Trade, mark and settled funding tables, manifest / 成交、标记价、已结算资金费、清单 |
| `crypto_perpetual_scale` | 64 contracts, 2025 hourly history / 64 合约、2025 小时历史 | Same three tables plus two transition proofs / 同上三表及两份周期变更证据 |

From the repository root on Linux, verify the exact snapshot bytes:

在 Linux 仓库根目录核对快照字节：

```bash
sha256sum --check data/SHA256SUMS
```

The four market sections in `config/config.ini` already point to the bundled integration datasets. Follow the [build and cloud test instructions](../guides/research.md#development-and-validation); for this work, builds and tests run only on the authorized Colab PRO 6000. This publication changes no numerical code or test configuration. The same 18 data files were checked against the cloud-tested input archive; no new backtest is implied by this upload.

`config/config.ini` 的四个市场 section 已指向附带的集成测试数据。构建和测试入口见[研究指南](../guides/research.zh-CN.md#开发与验证)；当前任务的构建及测试仍只在获授权的 Colab PRO 6000 执行。本次上传不修改数值代码或测试配置，18 个文件已与云端验证所用输入包逐字节核对，不代表重新跑过一轮回测。

For the 64-contract study, set `[crypto_perpetual] data_dir = ./data/crypto_perpetual_scale` and set `symbols` to the complete ordered list in that directory's `manifest.json`. The supplied study dates remain 2025-07-01 through 2025-12-31, with history from 2025-01-01. The 64-contract data is additional benchmark input; the default integration fixture remains the three-contract dataset.

使用 64 合约时，将 `[crypto_perpetual] data_dir` 改为 `./data/crypto_perpetual_scale`，并将 `symbols` 设置为该目录 `manifest.json` 中完整的有序列表。研究日期仍为 2025-07-01 至 2025-12-31，历史起点为 2025-01-01。64 合约快照供规模测试使用，默认集成测试仍使用三合约快照。

## Provenance and scope / 来源与范围

- US data is the existing legacy vendor artifact; its manifest does not establish the original provider, publication times or revision vintages. A-share metadata identifies Tencent raw/hfq daily history and the missing historical announcement/revision timestamps. / 美股是既有供应商历史工件，清单不足以确认原供应商及发布时间、修订版本；A 股清单标明腾讯原始/hfq 日线来源及公告时间、修订版本的缺口。
- Spot data derives from the preserved Quantiacs daily panel. The source path/hash remain recorded in the manifest; loading this extracted snapshot does not require the larger original archive. Historical Quantiacs files remain preserved separately. / 现货提取自保留的 Quantiacs 日线面板，清单保留原始路径及摘要；加载此快照无需完整原始归档，其他 Quantiacs 历史文件仍保留。
- Perpetual manifests retain Binance official archive URLs and checksums. ZIP archives are acquisition records, not loading dependencies, and are not duplicated here. Funding-transition JSON proofs are loading dependencies and are included. / 永续清单保留 Binance 官方归档 URL 和校验和；原始 ZIP 属于采集记录，加载无需使用，因此不重复纳入。周期变更 JSON 是加载所需证据，已纳入。
- These are fixed research fixtures, not complete vendor PIT histories or a guarantee of out-of-sample performance. The software MIT license does not replace upstream market-data terms. / 这些是固定研究测试快照，不代表完整供应商 PIT 或样本外盈利证据；软件 MIT 许可不替代上游行情条款。

Private portfolios, checkpoints, credentials and additional raw archives remain outside this published fixture set. / 私有组合、checkpoint、凭据和其他原始归档不属于这组已发布测试快照。

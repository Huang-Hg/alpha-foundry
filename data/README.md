# Sample Datasets

Offline sample datasets used when `foundry.open(...)` is called without a custom `data_dir`. All files are committed directly in Git (no Git LFS or network access needed).

| Directory | Preset | Coverage | Files |
| --- | --- | --- | --- |
| `us_equity_daily` | `market="us_equity"` | 500 US stocks, `2025-07-01` – `2026-06-30` (with prior lookback rows) | `panel.parquet`, `membership.parquet`, `manifest.json` |
| `a_share_pit` | `market="a_share"` | 365 China A-shares, `2024-03-01` – `2026-06-30` | `panel.parquet`, `membership.parquet`, `manifest.json` |
| `crypto_spot_daily` | `market="crypto_spot"` | 73 spot pairs, daily UTC bars through `2025-12-31` | `panel.parquet`, `manifest.json` |
| `crypto_perpetual` | `market="crypto_perpetual"` | `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, 2025 hourly bars & funding | `trade.parquet`, `mark.parquet`, `funding.parquet`, `manifest.json` |
| `crypto_perpetual_scale` | `market="crypto_perpetual"` (opt-in) | 64 USDT linear perpetuals, 2025 hourly bars & funding | `trade.parquet`, `mark.parquet`, `funding.parquet`, `manifest.json` |

`foundry.open(market="crypto_perpetual")` loads the 3-symbol dataset by default; pass `data_dir="./data/crypto_perpetual_scale"` to use the 64-symbol dataset:

```python
import foundry

with foundry.open(market="crypto_perpetual", data_dir="./data/crypto_perpetual_scale") as session:
    report = session.evaluate("cs_rank(ret_20)")
```

Each dataset directory includes a `manifest.json` describing its schema and field conventions. You can verify file checksums from the repository root with:

```bash
sha256sum --strict --check data/SHA256SUMS
```

To convert your own CSVs or database exports into Parquet datasets, see [guides/data.md](../guides/data.md) ([中文说明](../guides/data.zh-CN.md)).

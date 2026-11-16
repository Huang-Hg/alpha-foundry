# Bring your own market data

[Overview](../README.md) · [Research guide](research.md) · [简体中文](data.zh-CN.md)

Foundry reads local artifacts selected by `market`, `data_dir`, `start` and `end`. Your vendor, CSV export or database supplies these artifacts; the included snapshots are optional examples. Source data is truncated at the requested cutoff before features and execution rules are derived. Preserve real availability times, historical eligibility and earlier rolling history.

Choose one of two formats: daily equity/spot tables, or explicit crypto base bars. For stateful crypto programs, base bars make execution, decision and statistics frequencies explicit.

## Daily equity and spot tables

Write `panel.parquet` in long form, with one date/symbol row except for explicitly versioned A-share bars.

| Column/file | Contract |
| --- | --- |
| `date` | Midnight, timezone-naive exchange session date; UTC calendar date for spot |
| `symbol` | String identifier; preserve leading zeros |
| `open`, `high`, `low`, `close`, `volume` | Numeric OHLCV; absent observations remain missing |
| US `adj_factor` | OHLC adjustment factor; volume is unchanged |
| A-share `published` | Session date when that vintage was available |
| A-share `exec_open`, `exec_close` | Historically adjusted execution prices; raw OHLCV produces features |
| Spot `eligible` | Boolean historical eligibility at that close |
| Equity `membership.parquet` | `symbol`, `start`, `end`; inclusive start, exclusive end, null end means open-ended |
| A-share/spot `manifest.json` | Source contract described below |

For US equities, supply raw OHLC with factors, or already adjusted OHLC with factors equal to one. Do not adjust volume implicitly. Historical membership is separate from quotes: retain quotes outside eligibility so existing positions can still be valued or exited. Membership intervals must describe what was known then, rather than today's constituents.

The A-share reader selects the latest bar vintage whose `published <= date`. A late revision cannot replace an earlier decision bar. This is session-date publication handling; it does not establish intraday announcement times. Leave unavailable prices missing and disclose missing revisions or announcement history.

A complete US CSV conversion, assuming the CSVs already contain the columns above:

```python
from pathlib import Path
import pandas as pd

folder = Path("/datasets/my_us_equity")
folder.mkdir(parents=True, exist_ok=True)
panel = pd.read_csv("ohlcv.csv", parse_dates=["date"], dtype={"symbol": str})
members = pd.read_csv("membership.csv", parse_dates=["start", "end"], dtype={"symbol": str})
panel.to_parquet(folder / "panel.parquet", index=False)
members.to_parquet(folder / "membership.parquet", index=False)
```

```python
import foundry
with foundry.open(
    market="us_equity", data_dir="/datasets/my_us_equity",
    start="2025-07-01", end="2025-12-31", backend="cuda",
) as session:
    print(session.grid)
    report = session.evaluate("gt(ma_dist_20,0)")
```

Keep earlier history in the panel for rolling windows; `start` selects the research interval, not the first stored observation. Daily equity closes use exchange calendars, including DST and early closes; execution is no earlier than the next session open. A date-only end includes that session/day.

### Source metadata

A-share and daily spot manifests contain these seven string fields:

```json
{
  "schema": "my-daily-v1",
  "source": "Provider, export version and acquisition method",
  "retrieved_at": "Actual acquisition timestamp",
  "features": "Raw/adjusted feature-price convention",
  "execution_prices": "Adjustment and execution-price convention",
  "publication": "Publication timestamps and missing vintage limitations",
  "membership": "Historical eligibility source and availability limitations"
}
```

Use meaningful values for your data. The current US reader reports its fixed legacy provenance contract; placing a custom manifest next to US tables does not override that contract. Preserve your source record alongside them. Supporting another US provenance contract requires extending the reader. Neither missing metadata nor externally supplied cutoffs can certify a full vendor PIT history.

Additional Parquet columns do not automatically become formula fields. Use the published field list from `session.describe()`. To use arbitrary features/model logic without extending the market reader, compute causal scores/targets in your own application and register them against `session.grid`.

## Explicit crypto base bars

Both crypto spot and isolated USDT perpetuals accept `schema="crypto-bars-v1"`. Supply `bars.parquet` and `manifest.json`; perpetuals also need dense `funding.parquet`.

| File | Required columns |
| --- | --- |
| `bars.parquet` | String `symbol`; int64 UTC ns `open_ns`, `close_ns`, `available_ns`; floating `open`, `high`, `low`, `close`, `volume`; bool `eligible` |
| Perpetual bar additions | Floating `mark_open`, `mark_high`, `mark_low`, `mark_close` |
| `funding.parquet` | String `symbol`; int64 UTC ns `settlement_ns`; floating `rate`, `mark` |

Each base bar is the half-open interval `[open_ns,close_ns)`. Times align to the declared fixed `interval`; `close_ns = open_ns + interval`. `available_ns >= close_ns` is when the complete row and eligibility became visible. Duplicate `(symbol,open_ns)` observations are allowed only as distinct publication vintages with different `available_ns`.

Prices/marks are quote currency per asset unit; volume and order quantities use asset units. Funding `rate` is a notional fraction per settlement. All assets in an account use one consistent quote currency; the model does not perform cross-currency collateral conversion.

OHLC must be finite positive with consistent bounds, or all missing for a missing spot quote. Volume must be nonnegative and have the same missingness as the quote. Perpetual execution requires complete finite trade and mark bars. Spot can have missing quotes; they cannot fill orders, and the order model carries the last valid close for valuation, with zero before first listing.

The source is truncated before feature derivation. For each completed decision bar, only vintages available by that decision are used; late constituents make that feature bar unavailable instead of being backfilled from the future. Execution uses the first published factual base-bar vintage visible by the cutoff. Mark prices and funding remain execution-only inputs, excluded from the feature panel.

### Convert a publication-aware CSV

Assume `bars.csv` contains `open_time`, `close_time`, `available_time` as actual UTC-aware timestamps, plus the OHLCV/eligibility fields above. This spot example writes a complete manifest and actual artifact checksum:

```python
from pathlib import Path
import hashlib
import json
import pandas as pd

folder = Path("/datasets/crypto_5m")
folder.mkdir(parents=True, exist_ok=True)
bars = pd.read_csv("bars.csv", dtype={"symbol": str, "eligible": bool})
for source, target in (("open_time", "open_ns"), ("close_time", "close_ns"),
                       ("available_time", "available_ns")):
    stamps = pd.to_datetime(bars.pop(source), utc=True)
    bars[target] = stamps.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")
for name in ("open", "high", "low", "close", "volume"):
    bars[name] = bars[name].astype("float64")
bars.to_parquet(folder / "bars.parquet", index=False)


def iso_ns(value):
    return pd.Timestamp(int(value), tz="UTC").isoformat().replace("+00:00", "Z")


payload = (folder / "bars.parquet").read_bytes()
manifest = {
    "schema": "crypto-bars-v1", "market": "crypto_spot", "timezone": "UTC",
    "interval": "5min", "history_start": iso_ns(bars.open_ns.min()),
    "end": iso_ns(bars.close_ns.max()), "symbols": sorted(bars.symbol.unique().tolist()),
    "source": "My archived feed and export version",
    "retrieved_at": "Replace with actual acquisition timestamp",
    "features": "Trade OHLCV, frozen from published vintages",
    "execution_prices": "First published trade OHLC",
    "publication": "Actual available_time; disclose missing historical vintages",
    "membership": "Historical eligible values published with each complete bar",
    "artifacts": {"bars": {"path": "bars.parquet", "sha256": hashlib.sha256(payload).hexdigest()}},
}
(folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

Select `start`/`end` inside the stored coverage. Source history start, research start and cutoff must contain whole statistics periods. For daily statistics, prepare full UTC days. For an 8-hour source span, use a dividing statistics period such as `1h`, rather than claiming daily statistics. Do not invent publication timestamps: if only current revised prices exist, record that limitation instead of treating them as a complete historical publication archive.

```python
import foundry
with foundry.open(
    market="crypto_spot", data_dir="/datasets/crypto_5m",
    start="2025-01-08", end="2025-01-31", backend="cuda",
    clock=foundry.Clock(decision="15min", statistics="1D"),
    execution=foundry.Execution(model="orders"),
) as session:
    print(session.describe()["execution_plan"])
    report = session.evaluate("gt(ma_dist_20,0)")
```

Fixed durations use `s`, `min`, `h` or `D`; decisions and statistics must be integer multiples of the source base interval. Manifest bounds are explicit ISO UTC timestamps with exclusive `end`. A research end given as a timestamp is exclusive; a date-only end includes that UTC day. Choose research bounds aligned to statistics periods. Decision windows count completed decision bars.

### Add perpetual marks and funding

For `market="crypto_perpetual"`:

1. Include the four mark OHLC columns for every base-open/asset pair, matching the trade grid.
2. Write `funding.parquet` with one row for every base-open/asset pair. `settlement_ns` equals the base open, `rate=0` explicitly means no cashflow there, and `mark` is a finite positive settlement mark. Include actual nonzero settlements; missing data is not interpreted as zero.
3. Add `"funding": {"path": "funding.parquet", "sha256": "..."}` to `artifacts`, using the hash of the actual file, and set manifest `market` accordingly.
4. Open with `Account(margin="isolated", leverage=..., gross_limit=...)`. Leverage controls collateral requirements; gross limit is a separate exposure budget.

If settlements fall between base opens, provide finer base bars. The reader does not move settlements to a nearby timestamp. It also does not infer funding, mark extremes or liquidation paths from daily trade OHLC alone.

### Preserved hourly archive format

The included perpetual datasets use the separate archived `trade.parquet`, `mark.parquet`, `funding.parquet`, manifest and transition-proof format. It provides daily decisions/statistics with hourly execution and disclosed funding proxies. To acquire this format from supported public archives:

```python
import foundry
foundry.import_perpetual(
    "/datasets/perpetual_archive", symbols="BTCUSDT ETHUSDT SOLUSDT",
    history_start="2025-01-01", start="2025-07-01", end="2025-12-31",
)
```

The importer retains source URLs, checksums and required transition evidence. It does not generate `crypto-bars-v1`; use the explicit format above for your own finer-frequency exports. Existing dataset manifests determine asset order; the import `symbols` argument selects a new acquisition.

## Data, predictions and PIT

Market bars, decision features, execution quotes and model predictions have different roles. Daily close decisions cannot execute at the same day's earlier open. Eligibility controls new tradable cross-sections; quotes remain available for existing-position valuation. Model observation/fitting cutoffs include training-label information and preprocessing. `foundry.walk_forward` supplies causal feature prefixes, but callers remain responsible for independently loaded information and pretrained model provenance.

After opening, inspect `session.grid`, `session.describe()["clock"]`, fields and provenance before registering predictions. Follow [array registration](research.md#arrays-and-model-outputs) for exact `(B,D,S)` coordinates and `(B,D)` cutoffs. A data contract records what is known and missing; it is not a guarantee of out-of-sample performance.

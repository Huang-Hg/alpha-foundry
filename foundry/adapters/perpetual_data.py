"""Binance public USD-M archives and their explicit hourly research projection.

The importer verifies official SHA256 checksums, retains source bytes and records
retrieval time. A completed output directory is immutable: a new revision needs
a new output path. The loader performs no network access.
"""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from foundry.adapters.files import publish
from foundry.contracts import DataContract, MarketData
from foundry.perpetual_types import PerpetualData


BASE = 'https://data.binance.vision/data/futures/um/monthly'
HOUR_MS = 3_600_000
SCHEMA = 'binance-um-hourly-v1'
KINDS = {'trade': 'klines', 'mark': 'markPriceKlines', 'funding': 'fundingRate'}
COLUMNS = {'trade': ('open_time', 'open', 'high', 'low', 'close', 'volume'),
           'mark': ('open_time', 'open', 'high', 'low', 'close'),
           'funding': ('calc_time', 'funding_interval_hours', 'last_funding_rate')}


def archive_url(kind, symbol, month):
    dataset = KINDS[kind]
    folder = f'{dataset}/{symbol}'
    if kind == 'funding':
        filename = f'{symbol}-fundingRate-{month}.zip'
    else:
        folder += '/1h'
        filename = f'{symbol}-1h-{month}.zip'
    return f'{BASE}/{folder}/{filename}'


def _verified_archive(raw, kind, symbol, month):
    url = archive_url(kind, symbol, month)
    filename = url.rsplit('/', 1)[1]
    directory = raw / KINDS[kind] / symbol
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / filename
    checksum = directory / (filename + '.CHECKSUM')
    receipt = directory / (filename + '.json')
    cached = receipt.exists()
    if cached:
        record = json.loads(receipt.read_text(encoding='utf-8'))
        payload, proof = archive.read_bytes(), checksum.read_bytes()
        if record['url'] != url:
            raise ValueError(f'cached source URL differs: {receipt}')
    else:
        with urlopen(url + '.CHECKSUM', timeout=60) as response:
            proof = response.read()
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        record = dict(url=url, checksum_url=url + '.CHECKSUM',
                      retrieved_at=datetime.now(timezone.utc).isoformat())
    tokens = proof.decode('ascii').split()
    if len(tokens) != 2 or tokens[1].lstrip('*') != filename:
        raise ValueError(f'invalid official checksum record: {url}')
    digest = hashlib.sha256(payload).hexdigest()
    if digest != tokens[0].lower():
        raise ValueError(f'SHA256 verification failed: {url}')
    if cached and record['sha256'] != digest:
        raise ValueError(f'cached source differs from its retrieval receipt: {archive}')
    record.update(sha256=digest, bytes=len(payload), kind=kind, symbol=symbol, month=month,
                  archive=archive.relative_to(raw.parent).as_posix(),
                  checksum=checksum.relative_to(raw.parent).as_posix())
    if not cached:
        publish(archive, payload)
        publish(checksum, proof)
        publish(receipt, json.dumps(record, indent=2).encode())
    with ZipFile(BytesIO(payload)) as packed:
        if len(packed.namelist()) != 1 or not packed.namelist()[0].endswith('.csv'):
            raise ValueError(f'expected one CSV in archive: {url}')
        with packed.open(packed.namelist()[0]) as source:
            frame = pd.read_csv(source, usecols=list(COLUMNS[kind]))
    frame.insert(0, 'symbol', symbol)
    return frame, record


def import_perpetual(output, cfg):
    """Import configured monthly archives; do not overwrite a published dataset."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'published dataset is immutable: {output}')
    start, end = pd.Timestamp(cfg.history_start), pd.Timestamp(cfg.end)
    symbols = tuple(cfg.symbols.split())
    if start > end or start != start.normalize() or end != end.normalize():
        raise ValueError('perpetual history bounds must be ordered UTC dates')
    if not symbols or len(set(symbols)) != len(symbols) or any(not s.isalnum() for s in symbols):
        raise ValueError('perpetual symbols must be unique alphanumeric identifiers')
    frames = {kind: [] for kind in KINDS}
    sources = []
    months = pd.period_range(start, end, freq='M')
    jobs = [(kind, symbol, str(month)) for symbol in symbols for month in months for kind in KINDS]
    jobs.extend(('funding', symbol, str(months[-1] + 1)) for symbol in symbols)
    with ThreadPoolExecutor(max_workers=8) as workers:
        imported = workers.map(lambda job: _verified_archive(output / 'raw', *job), jobs)
        for frame, record in imported:
            frames[record['kind']].append(frame)
            sources.append(record)
            print(f'[{len(sources)}/{len(jobs)}] verified {record["url"]}', flush=True)
    contract = DataContract(SCHEMA, 'Binance official public USD-M monthly archives; SHA256 verified',
        max(source['retrieved_at'] for source in sources),
        'Completed UTC daily trade OHLCV aggregated from complete hourly bars; funding and mark prices are execution-only.',
        'Hourly trade open and mark OHLC. Settlement mark is absent from funding CSV: hourly mark open is an explicit proxy; actual calc_time is retained and bucketed into its UTC hour.',
        'Public archive retrieval vintage only; historical publication/revision vintages unavailable. Daily features are known at UTC day close. Funding is realized execution cashflow, never a feature. Constant-interval coverage includes the next settlement timestamp without using its rate; shortened interval-change gaps require an archived funding-history API confirmation.',
        'Fixed preselected configured assets across the imported interval; not a reconstructed historical dynamic universe.')
    tables, artifacts = {}, {}
    lower, upper = start.value // 1_000_000, (end + pd.Timedelta(days=1)).value // 1_000_000
    for kind, parts in frames.items():
        time = 'calc_time' if kind == 'funding' else 'open_time'
        table = pd.concat(parts, ignore_index=True)
        table = table.sort_values(['symbol', time])
        inside = table[(table[time] >= lower) & (table[time] < upper)]
        if kind == 'funding':
            successor = table[table[time] >= upper].groupby('symbol', sort=False).head(1)
            table = pd.concat((inside, successor), ignore_index=True).sort_values(['symbol', time])
        else:
            table = inside
        tables[kind] = table
        payload = table.to_parquet(index=False)
        artifacts[kind] = dict(path=kind + '.parquet', sha256=hashlib.sha256(payload).hexdigest(), rows=len(table))
        publish(output / artifacts[kind]['path'], payload)
    transitions = _verify_transitions(output, tables['funding'])
    sources.extend(transitions)
    manifest = dict(asdict(contract), history_start=cfg.history_start, end=cfg.end, symbols=list(symbols),
                    timezone='UTC', interval='1h', sources=sources, artifacts=artifacts,
                    built_at=datetime.now(timezone.utc).isoformat())
    # Validate the whole requested span before making its manifest visible.
    _project(tables, manifest, cfg.end)
    publish(output / 'manifest.json', json.dumps(manifest, indent=2).encode())
    return manifest


def _hourly_grid(frame, assets, times, columns, name):
    if frame.duplicated(['symbol', 'open_time']).any():
        raise ValueError(f'{name} contains duplicate asset/hour rows')
    if set(frame['symbol']) != set(assets) or len(frame) != len(times) * len(assets):
        raise ValueError(f'{name} does not cover the complete configured asset/hour grid')
    if not np.isin(frame['open_time'].to_numpy(), times).all():
        raise ValueError(f'{name} contains non-hourly timestamps')
    grids = {column: np.ascontiguousarray(frame.pivot(index='open_time', columns='symbol', values=column)
             .reindex(index=times, columns=assets).to_numpy(dtype=np.float64)) for column in columns}
    prices = np.stack([grids[key] for key in ('open', 'high', 'low', 'close')])
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f'{name} has missing or invalid hourly prices')
    if ((grids['high'] < np.maximum(grids['open'], grids['close'])) |
        (grids['low'] > np.minimum(grids['open'], grids['close']))).any():
        raise ValueError(f'{name} OHLC bounds are inconsistent')
    return grids


def _transition_candidates(funding):
    """Only a shortened gap at an interval change can be a transition."""
    for symbol, events in funding.groupby('symbol', sort=False):
        events = events.sort_values('calc_time')
        times = events.calc_time.to_numpy(dtype=np.int64)
        hours = times // HOUR_MS
        spans = events.funding_interval_hours.to_numpy()
        for i in np.flatnonzero((np.diff(hours) < spans[1:]) & (spans[1:] != spans[:-1])):
            yield symbol, int(times[i]), int(times[i+1])


def _verify_transitions(output, funding):
    """Resolve ambiguous frequency changes against the independent REST history."""
    records = []
    for symbol, first, last in _transition_candidates(funding):
        url = (f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}'
               f'&startTime={first}&endTime={last}&limit=1000')
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        events = json.loads(payload)
        if [(event['symbol'], event['fundingTime']) for event in events] != [(symbol, first), (symbol, last)]:
            raise ValueError(f'funding interval transition has missing or inconsistent settlements: {symbol}')
        path = f'raw/transitions/{symbol}-{first}-{last}.json'
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        publish(output / path, payload)
        records.append(dict(kind='funding_transition', symbol=symbol, first=first, last=last,
                            url=url, archive=path, sha256=hashlib.sha256(payload).hexdigest(),
                            bytes=len(payload), retrieved_at=datetime.now(timezone.utc).isoformat()))
    return records


def _funding_coverage(funding, successors, assets, lower, upper, transitions):
    """Constant intervals prove adjacency; API evidence resolves frequency changes."""
    if set(successors['symbol']) != set(assets):
        raise ValueError('funding requires the next settlement timestamp/interval to prove tail coverage')
    proof = pd.concat((funding[['symbol', 'calc_time', 'funding_interval_hours']], successors), ignore_index=True)
    if not np.isfinite(proof[['calc_time', 'funding_interval_hours']].to_numpy()).all():
        raise ValueError('funding coverage timestamps/intervals must be finite')
    if not pd.api.types.is_integer_dtype(proof['calc_time'].dtype):
        raise ValueError('funding settlement timestamps must be integer UTC milliseconds')
    intervals = proof['funding_interval_hours'].to_numpy()
    if ((intervals <= 0) | (intervals != np.floor(intervals))).any():
        raise ValueError('funding settlement intervals must be positive whole hours')
    for asset in assets:
        events = proof[proof['symbol'] == asset].sort_values('calc_time')
        hours = events['calc_time'].to_numpy(dtype=np.int64) // HOUR_MS * HOUR_MS
        spans = events['funding_interval_hours'].to_numpy(dtype=np.int64) * HOUR_MS
        if hours[0] >= upper or hours[0] - spans[0] >= lower:
            raise ValueError(f'funding has a missing initial settlement: {asset}')
        times = events['calc_time'].to_numpy(dtype=np.int64)
        for i in np.flatnonzero(np.diff(hours) != spans[1:]):
            transition = (asset, int(times[i]), int(times[i+1]))
            if not (spans[i] != spans[i+1] and 0 < hours[i+1]-hours[i] < spans[i+1]
                    and transition in transitions):
                raise ValueError(f'funding settlement coverage has a gap or inconsistent interval: {asset}')


def _project(tables, manifest, asof):
    """Truncate observations before any daily aggregation or event mapping."""
    if manifest['schema'] != SCHEMA or manifest['timezone'] != 'UTC' or manifest['interval'] != '1h':
        raise ValueError('unsupported perpetual dataset contract')
    first, last = pd.Timestamp(manifest['history_start']), pd.Timestamp(asof)
    if last < first or last > pd.Timestamp(manifest['end']) or last != last.normalize():
        raise ValueError('perpetual asof must be a complete UTC day inside the imported span')
    lower, upper = first.value // 1_000_000, (last + pd.Timedelta(days=1)).value // 1_000_000
    times = np.arange(lower, upper, HOUR_MS, dtype=np.int64)
    dates = np.arange(np.datetime64(first.date()), np.datetime64(last.date()) + 1, dtype='datetime64[D]')
    assets = tuple(manifest['symbols'])
    cut = {}
    for kind, table in tables.items():
        time = 'calc_time' if kind == 'funding' else 'open_time'
        cut[kind] = table[(table[time] >= lower) & (table[time] < upper)]
    trade = _hourly_grid(cut['trade'], assets, times, ('open', 'high', 'low', 'close', 'volume'), 'trade')
    mark = _hourly_grid(cut['mark'], assets, times, ('open', 'high', 'low', 'close'), 'mark')
    if not np.isfinite(trade['volume']).all() or (trade['volume'] < 0).any():
        raise ValueError('trade has missing or negative hourly volume')
    funding = cut['funding'].copy()
    if set(funding['symbol']) != set(assets):
        raise ValueError('funding observations must cover every configured asset')
    if not np.isfinite(funding['last_funding_rate'].to_numpy()).all():
        raise ValueError('funding contains nonfinite observations')
    funding['hour'] = funding['calc_time'] // HOUR_MS * HOUR_MS
    if funding.duplicated(['symbol', 'hour']).any():
        raise ValueError('multiple funding settlements in an asset/hour require a finer event model')
    # Only successor metadata enters this proof. Its future realized rate never
    # affects a historical execution array, feature or risk report.
    future = tables['funding'].loc[tables['funding']['calc_time'] >= upper,
                                  ['symbol', 'calc_time', 'funding_interval_hours']]
    successors = future.sort_values('calc_time').groupby('symbol', sort=False).head(1)
    transitions = {(s['symbol'], s['first'], s['last']) for s in manifest['sources']
                   if s['kind'] == 'funding_transition'}
    _funding_coverage(funding, successors, assets, lower, upper, transitions)
    rates = np.zeros((len(times), len(assets)), dtype=np.float64)
    asset_index = {asset: i for i, asset in enumerate(assets)}
    rows = ((funding['hour'].to_numpy(dtype=np.int64) - lower) // HOUR_MS)
    cols = np.array([asset_index[symbol] for symbol in funding['symbol']], dtype=np.int64)
    rates[rows, cols] = funding['last_funding_rate'].to_numpy(dtype=np.float64)
    cube = {key: value.reshape(len(dates), 24, len(assets)) for key, value in trade.items()}
    daily = dict(open=cube['open'][:, 0], high=cube['high'].max(1), low=cube['low'].min(1),
                 close=cube['close'][:, -1], volume=cube['volume'].sum(1))
    contract = DataContract(**{key: manifest[key] for key in DataContract.__dataclass_fields__})
    data = MarketData(dates, assets, {key: np.ascontiguousarray(value) for key, value in daily.items()},
                      np.ones((len(dates), len(assets)), dtype=bool), contract)
    return PerpetualData(data, times, np.repeat(np.arange(len(dates), dtype=np.int32), 24),
                         trade['open'], mark['open'], mark['high'], mark['low'], mark['close'],
                         rates, mark['open'].copy())


def load_perpetual(directory, study):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    upper = (pd.Timestamp(study.asof) + pd.Timedelta(days=1)).value // 1_000_000
    for source in manifest['sources']:
        if source['kind'] == 'funding_transition' and source['first'] < upper:
            symbol, first, last = source['symbol'], source['first'], source['last']
            path = f'raw/transitions/{symbol}-{first}-{last}.json'
            if source['archive'] != path:
                raise ValueError('funding transition path differs from canonical evidence path')
            payload = (directory / path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != source['sha256']:
                raise ValueError('funding transition SHA256 differs')
            events = json.loads(payload)
            if [(e['symbol'], e['fundingTime']) for e in events] != [(symbol, first), (symbol, last)]:
                raise ValueError('funding transition evidence contains missing or inconsistent settlements')
    tables = {}
    for kind in KINDS:
        artifact = manifest['artifacts'][kind]
        if artifact['path'] != kind + '.parquet':
            raise ValueError('perpetual artifact path must name its local canonical table')
        payload = (directory / artifact['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact['sha256']:
            raise ValueError(f'perpetual artifact SHA256 differs: {kind}')
        tables[kind] = pd.read_parquet(BytesIO(payload))
    return _project(tables, manifest, study.asof)

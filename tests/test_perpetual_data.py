"""Hourly artifact/PIT boundaries from local synthetic parquet, without network."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from foundry.adapters.perpetual_data import load_perpetual
from foundry.contracts import DataContract, Study
from foundry.market.crypto_perpetual import CryptoPerpetual
from foundry.runtime import prepare_backend


FIRST_MS = 1735862400000  # 2025-01-03 00:00 UTC (Friday)
HOUR_MS = 3_600_000
ASSETS = ('BETAUSDT', 'ALPHAUSDT')
CONTRACT = DataContract('binance-um-hourly-v1', 'synthetic hourly fixture', 'test-fixture',
    'completed UTC daily trade OHLCV; funding and mark are execution-only',
    'hourly mark-open settlement proxy, actual funding calc_time retained in artifact',
    'publication and historical revision vintages unavailable',
    'fixed preselected assets, not a historical dynamic universe')


def tables_fixture():
    trade, mark, funding = [], [], []
    for index, symbol in enumerate(ASSETS):
        for hour in range(72):
            time, price = FIRST_MS + hour * HOUR_MS, 100. + index * 100. + hour / 100.
            trade.append(dict(symbol=symbol, open_time=time, open=price, high=price+.01,
                              low=price-.01, close=price+.005, volume=hour+1.))
            mark.append(dict(symbol=symbol, open_time=time, open=price*1.001,
                             high=(price+.01)*1.001, low=(price-.01)*1.001, close=(price+.005)*1.001))
            if hour % 8 == 0:
                funding.append(dict(symbol=symbol, calc_time=time+123, funding_interval_hours=8,
                                    last_funding_rate=.0001*(index+1)))
        # The first subsequent settlement proves coverage through the requested last UTC day.
        funding.append(dict(symbol=symbol, calc_time=FIRST_MS+72*HOUR_MS+123, funding_interval_hours=8,
                            last_funding_rate=.0001*(index+1)))
    return dict(trade=pd.DataFrame(trade), mark=pd.DataFrame(mark), funding=pd.DataFrame(funding))


def write_dataset(root, tables):
    artifacts = {}
    for kind, table in tables.items():
        payload = table.to_parquet(index=False)
        (root/f'{kind}.parquet').write_bytes(payload)
        artifacts[kind] = dict(path=f'{kind}.parquet', sha256=hashlib.sha256(payload).hexdigest(), rows=len(table))
    manifest = dict(asdict(CONTRACT), history_start='2025-01-03', end='2025-01-05',
                    symbols=list(ASSETS), timezone='UTC', interval='1h', sources=[], artifacts=artifacts)
    (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


class PerpetualDataTests(unittest.TestCase):
    def test_complete_utc_days_keep_weekends_order_and_funding_settlement_proxy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_dataset(root, tables_fixture())
            data = load_perpetual(root, Study('crypto_perpetual', '2025-01-04', '2025-01-05'))
            np.testing.assert_array_equal(data.daily.dates, np.arange('2025-01-03', '2025-01-06', dtype='datetime64[D]'))
            self.assertEqual(data.daily.assets, ASSETS)
            np.testing.assert_array_equal(data.event_times_ms, FIRST_MS + np.arange(72)*HOUR_MS)
            np.testing.assert_array_equal(data.day_index, np.repeat(np.arange(3), 24))
            np.testing.assert_allclose(data.daily.values['open'][:, 0], [100., 100.24, 100.48], rtol=0, atol=1e-13)
            np.testing.assert_allclose(data.daily.values['close'][:, 0], [100.235, 100.475, 100.715], rtol=0, atol=1e-13)
            np.testing.assert_allclose(data.daily.values['volume'][:, 0], [300., 876., 1452.], rtol=0, atol=0)
            np.testing.assert_allclose(data.funding_rate[::8], np.tile([.0001, .0002], (9, 1)), rtol=0, atol=0)
            np.testing.assert_array_equal(data.funding_rate[np.arange(72) % 8 != 0], 0.)
            np.testing.assert_array_equal(data.funding_mark, data.mark_open)
            self.assertEqual(data.daily.contract, CONTRACT)
            self.assertIn('unavailable', data.daily.contract.publication)
            self.assertIn('proxy', data.daily.contract.execution_prices)

    def test_asof_truncation_precedes_daily_aggregation_and_future_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tables = tables_fixture()
            write_dataset(root, tables)
            study = Study('crypto_perpetual', '2025-01-03', '2025-01-04')
            before = load_perpetual(root, study)
            for kind in ('trade', 'mark'):
                future = tables[kind].open_time >= FIRST_MS + 48*HOUR_MS
                tables[kind].loc[future, ['open', 'high', 'low', 'close']] = np.nan
            tables['funding'].loc[tables['funding'].calc_time >= FIRST_MS + 48*HOUR_MS, 'last_funding_rate'] = 100.
            write_dataset(root, tables)
            after = load_perpetual(root, study)
            for name in before.daily.values:
                np.testing.assert_array_equal(after.daily.values[name], before.daily.values[name])
            np.testing.assert_array_equal(after.funding_rate, before.funding_rate)
            np.testing.assert_array_equal(after.mark_close, before.mark_close)
            with self.assertRaisesRegex(ValueError, 'invalid hourly prices'):
                load_perpetual(root, Study(study.market, study.start, '2025-01-05'))

    def test_funding_and_mark_changes_do_not_enter_the_feature_panel(self):
        backend = prepare_backend('cpu')
        self.addCleanup(backend.close)
        study = Study('crypto_perpetual', '2025-01-03', '2025-01-05')
        market = CryptoPerpetual(5., .0004, .005, .0005)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tables = tables_fixture()
            write_dataset(root, tables)
            before = market.compile(load_perpetual(root, study), backend, study)
            tables['funding']['last_funding_rate'] *= -10
            tables['mark'][['open', 'high', 'low', 'close']] *= 1.01
            write_dataset(root, tables)
            after = market.compile(load_perpetual(root, study), backend, study)
            self.assertEqual(before.panel.snapshot, after.panel.snapshot)
            self.assertNotEqual(before.host_spec.snapshot, after.host_spec.snapshot)
            self.assertFalse(any('funding' in name or 'mark_' in name for name in after.panel.fields))
            for name in ('open', 'high', 'low', 'close', 'volume', 'candlebody', 'rangerel'):
                np.testing.assert_array_equal(after.panel[name], before.panel[name])
            self.assertEqual(after.window.start, 0)
            np.testing.assert_array_equal(np.flatnonzero(after.host_spec.rebalance), [0, 24, 48])
            before.panel.release()
            after.panel.release()

    def test_missing_hours_duplicate_funding_and_corrupt_artifacts_fail_at_io_boundary(self):
        study = Study('crypto_perpetual', '2025-01-03', '2025-01-05')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tables = tables_fixture()
            tables['mark'] = tables['mark'].drop(index=25)
            write_dataset(root, tables)
            with self.assertRaisesRegex(ValueError, 'complete configured asset/hour grid'):
                load_perpetual(root, study)
            tables = tables_fixture()
            tables['funding'] = pd.concat([tables['funding'], tables['funding'].iloc[:1]], ignore_index=True)
            write_dataset(root, tables)
            with self.assertRaisesRegex(ValueError, 'multiple funding settlements'):
                load_perpetual(root, study)
            write_dataset(root, tables_fixture())
            with (root/'trade.parquet').open('ab') as artifact:
                artifact.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_perpetual(root, study)

    def test_missing_funding_settlement_cannot_be_reinterpreted_as_zero_rate(self):
        study = Study('crypto_perpetual', '2025-01-03', '2025-01-05')
        for missing in (0, 1, 8):  # first, interior and last expected settlement for BETA
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                tables = tables_fixture()
                tables['funding'] = tables['funding'].drop(index=missing)
                write_dataset(root, tables)
                with self.assertRaisesRegex(ValueError, 'funding'):
                    load_perpetual(root, study)


if __name__ == '__main__':
    unittest.main()

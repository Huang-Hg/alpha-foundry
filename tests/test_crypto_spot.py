"""Synthetic daily crypto rules and reader boundaries; no private data or GPU."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from foundry.adapters.data import load_data
from foundry.backtest import cpu
from foundry.config import load_config
from foundry.contracts import DataContract, MarketData, Study
from foundry.market.crypto_spot import CryptoSpot
from foundry.runtime import prepare_backend


CONTRACT = DataContract('crypto-spot-daily-v1', 'synthetic legacy OHLCV and vendor eligibility fixture',
                        'test-fixture', 'unadjusted OHLCV', 'unadjusted daily OHLC',
                        'close availability assumed; publication timestamps and revision vintages unavailable',
                        'historical vendor eligibility flag; announcement timestamps unavailable')


def data_fixture(open_prices, close_prices):
    open_ = np.asarray(open_prices, dtype=np.float64)[:, None]
    close = np.asarray(close_prices, dtype=np.float64)[:, None]
    dates = np.arange(np.datetime64('2024-01-05'), np.datetime64('2024-01-05')+len(open_))
    values = dict(open=open_, close=close, high=np.maximum(open_, close)*1.01,
                  low=np.minimum(open_, close)*.99, volume=np.full_like(open_, 1000.))
    return MarketData(dates, ('COIN',), values, np.ones_like(open_, dtype=bool), CONTRACT)


class CryptoSpotTests(unittest.TestCase):
    def setUp(self):
        self.backend, self.executor = prepare_backend('cpu'), cpu.prepare()
        self.addCleanup(self.backend.close)
        self.addCleanup(self.executor.close)

    def compile(self, data, cost=0.):
        study = Study('crypto_spot', str(data.dates[0]), str(data.dates[-1]))
        return CryptoSpot(cost).compile(data, self.backend, study)

    def simulate(self, loaded, weights):
        return self.executor.simulate(np.asarray(weights, dtype=float)[None, :, None],
                                      self.executor.device_spec(loaded.host_spec), loaded.window.start,
                                      loaded.market.ppy, 'native')

    def test_reader_keeps_weekends_missing_days_and_historical_eligibility(self):
        rows = []
        for day, price, eligible in (('2024-01-05', 100., True), ('2024-01-06', 110., True),
                                     ('2024-01-08', 120., False), ('2024-01-09', 132., True)):
            rows.append(dict(date=pd.Timestamp(day), symbol='BASE', open=price, high=price+1,
                             low=price-1, close=price, volume=1000., eligible=eligible))
            # Placeholder rows must not make a future listing visible in old cross-sections.
            late = 200. if day == '2024-01-09' else np.nan
            rows.append(dict(date=pd.Timestamp(day), symbol='LATE', open=late, high=late,
                             low=late, close=late, volume=1000., eligible=day == '2024-01-09'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame(rows).to_parquet(root/'panel.parquet', index=False)
            (root/'manifest.json').write_text(json.dumps(asdict(CONTRACT)))
            settings = load_config()
            settings = replace(settings, crypto_spot=replace(settings.crypto_spot, data_dir=str(root)))
            short = load_data(settings, Study('crypto_spot', '2024-01-06', '2024-01-08'))
            full = load_data(settings, Study('crypto_spot', '2024-01-06', '2024-01-10'))
            np.testing.assert_array_equal(short.dates, np.arange('2024-01-05', '2024-01-09', dtype='datetime64[D]'))
            np.testing.assert_array_equal(full.dates, np.arange('2024-01-05', '2024-01-11', dtype='datetime64[D]'))
            self.assertEqual(short.assets, ('BASE',))
            self.assertEqual(full.assets, ('BASE', 'LATE'))
            self.assertTrue(np.isnan(full.values['open'][2]).all())
            self.assertTrue(np.isnan(full.values['open'][-1]).all())
            self.assertFalse(full.members[2].any())
            self.assertFalse(full.members[-1].any())
            self.assertFalse(short.members[3, 0])
            self.assertEqual(short.values['close'][3, 0], 120.)
            np.testing.assert_array_equal(short.values['close'], full.values['close'][:4, :1])
            self.assertEqual(full.contract, CONTRACT)
            self.assertIn('unavailable', full.contract.publication)
            self.assertIn('historical vendor', full.contract.membership)

    def test_close_signal_first_trades_saturday_open_after_the_price_gap(self):
        loaded = self.compile(data_fixture([100., 120., 132.], [100., 132., 132.]))
        self.assertEqual(loaded.market.ppy, 365.)
        self.assertTrue(loaded.market.geometric)
        result = self.simulate(loaded, [1., 0., 0.])
        returns, stats = result.returns, result.statistics
        np.testing.assert_allclose(returns[0], [0., .1, 0.], rtol=0, atol=1e-12)
        self.assertAlmostEqual(stats[0, 1], returns[0].mean()*365.)
        self.assertAlmostEqual(stats[0, 2], returns[0].std()*np.sqrt(365.))
        self.assertFalse(loaded.host_spec.block.any())
        self.assertIsNone(loaded.host_spec.carry)

    def test_missing_open_freezes_existing_holding_until_the_next_real_open(self):
        data = data_fixture([100., 100., np.nan, 120.], [100., 110., np.nan, 120.])
        loaded = self.compile(data)
        self.assertEqual(int(loaded.host_spec.block[2, 0]), 3)
        self.assertTrue(bool(loaded.host_spec.live[2, 0]))
        self.assertFalse(bool(loaded.market.universe(loaded.panel)[2, 0]))
        returns = self.simulate(loaded, [1., 0., 0., 0.]).returns
        np.testing.assert_allclose(returns[0], [0., .1, 0., 120./110.-1.], rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(np.prod(1+returns)), 1.2)
        prefix = MarketData(data.dates[:3].copy(), data.assets,
                            {name: value[:3].copy() for name, value in data.values.items()},
                            data.members[:3].copy(), data.contract)
        shorter = self.simulate(self.compile(prefix), [1., 0., 0.]).returns
        np.testing.assert_array_equal(shorter[0], returns[0, :3])

    def test_fee_and_open_eligibility_do_not_consult_same_day_range_or_volume(self):
        data = data_fixture([100., 110., 105., 115.], [100., 110., 105., 115.])
        fee = .001
        normal = self.compile(data, fee)
        values = {name: value.copy() for name, value in data.values.items()}
        values['high'][1:] *= 100
        values['low'][1:] *= .01
        values['close'][1:] = np.nan
        values['volume'][1:] = 0
        changed = self.compile(MarketData(data.dates.copy(), data.assets, values, data.members.copy(), CONTRACT), fee)
        np.testing.assert_array_equal(normal.host_spec.cost_buy, data.values['open']*fee)
        np.testing.assert_array_equal(changed.host_spec.cost_buy, normal.host_spec.cost_buy)
        np.testing.assert_array_equal(changed.host_spec.live, normal.host_spec.live)
        np.testing.assert_array_equal(changed.host_spec.block, normal.host_spec.block)
        self.assertIsNone(normal.host_spec.cost_sell)
        self.assertIsNone(normal.host_spec.carry)
        flat = data_fixture([100.]*3, [100.]*3)
        free = self.simulate(self.compile(flat), [1., 0., 0.]).returns
        paid = self.simulate(self.compile(flat, fee), [1., 0., 0.]).returns
        self.assertAlmostEqual(float(np.prod(1+free)), 1.)
        self.assertLess(float(np.prod(1+paid)), 1.)


if __name__ == '__main__':
    unittest.main()

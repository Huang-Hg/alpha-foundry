"""Hand-calculated isolated-margin accounts; run native CPU tests on Colab/CI."""
from dataclasses import fields, replace
import unittest

import numpy as np

from foundry.backtest import perpetual_cpu
from foundry.backtest.perpetual import MODEL, PerpetualSpec
from foundry.contracts import Allocation
from foundry.portfolio.allocation import cap_book, signal_to_book


def flat_spec(days=3, assets=1, fee=0., maintenance=.005):
    count = days * 24
    price = np.full((count, assets), 100.)
    return PerpetualSpec(
        1735689600000 + np.arange(count, dtype=np.int64) * 3600000,
        np.repeat(np.arange(days, dtype=np.int64), 24), np.arange(count) % 24 == 0,
        price, price, price, price, price, np.zeros_like(price), price,
        np.full_like(price, maintenance), np.full_like(price, fee),
        np.full_like(price, .0005), 5., days)


def target(spec, exposures):
    """One close decision on day zero, then flatten at the next daily rebalance."""
    book = np.zeros((len(exposures), *spec.shape))
    book[:, 0] = np.asarray(exposures, dtype=float)
    return book


def host_result(result, to_host):
    risk = replace(result.risk, **{f.name: to_host(getattr(result.risk, f.name))
                                  for f in fields(result.risk) if f.name != 'model'})
    return replace(result, returns=to_host(result.returns), statistics=to_host(result.statistics), risk=risk)


class HandCalculatedAccounts:
    """Shared assertions execute each implementation against independent arithmetic."""
    def run_book(self, spec, book, start=0):
        result = self.executor.simulate(self.xp.asarray(book), self.executor.device_spec(spec), start, 365., 'native')
        self.assertIsNotNone(result.risk)
        return host_result(result, self.to_host)

    def assert_event(self, risk, spec, event, flag, asset=0):
        expected = np.zeros_like(risk.liquidation_events)
        expected[:, event, asset] = flag
        np.testing.assert_array_equal(risk.liquidation_events, expected)
        np.testing.assert_array_equal(risk.first_liquidation_ms, spec.event_times_ms[event])
        np.testing.assert_array_equal(risk.liquidations.sum(1), 1)

    def test_zero_positions_have_no_cashflows_or_liquidations(self):
        spec = flat_spec()
        spec = replace(spec, funding_rate=np.full_like(spec.funding_rate, .9),
                       mark_low=np.full_like(spec.mark_low, 1.), mark_high=np.full_like(spec.mark_high, 1000.))
        result = self.run_book(spec, target(spec, [[0.]]))
        np.testing.assert_array_equal(result.returns, 0.)
        self.assertEqual(result.risk.model, MODEL)
        for name in ('funding', 'trading_fees', 'liquidation_fees', 'isolated_loss_cap',
                     'liquidations', 'liquidation_events', 'max_gross_leverage'):
            np.testing.assert_array_equal(getattr(result.risk, name), 0)
        np.testing.assert_array_equal(result.risk.first_liquidation_ms, -1)
        np.testing.assert_array_equal(result.risk.ending_equity, 1.)
        np.testing.assert_array_equal(result.risk.bankrupt, False)

    def test_funding_signs_and_new_entry_occurs_after_settlement(self):
        for rate in (-.01, .01):
            with self.subTest(rate=rate):
                spec = flat_spec(days=2)
                funding = spec.funding_rate.copy()
                funding[24] = .9       # no old quantity: opening after settlement pays nothing
                funding[25] = rate
                spec = replace(spec, funding_rate=funding)
                result = self.run_book(spec, target(spec, [[5.], [-5.]]))
                payment = np.array([-5. * rate, 5. * rate])
                np.testing.assert_allclose(result.returns[:, 1], payment, rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.risk.funding[:, 1], payment, rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.risk.ending_equity, 1 + payment, rtol=0, atol=1e-14)
                np.testing.assert_array_equal(result.risk.liquidations, 0)

    def test_signed_five_times_exposure_is_not_capped_to_cash_budget(self):
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                spec = flat_spec(days=2)
                close, high, low = spec.mark_close.copy(), spec.mark_high.copy(), spec.mark_low.copy()
                close[-1] = 100 + sign * 10
                high[-1], low[-1] = max(100., close[-1, 0]), min(100., close[-1, 0])
                spec = replace(spec, mark_close=close, mark_high=high, mark_low=low)
                result = self.run_book(spec, target(spec, [[sign * 5.]]))
                np.testing.assert_allclose(result.returns, [[0., .5]], rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.risk.ending_equity, 1.5, rtol=0, atol=1e-14)
                self.assertGreaterEqual(result.risk.max_gross_leverage[0], 5. - 1e-14)

    def test_round_trip_fees_are_self_financing(self):
        spec = flat_spec(fee=.01)
        book = target(spec, [[5.], [-5.]])
        result = self.run_book(spec, book)
        free = self.run_book(replace(spec, fee_rate=np.zeros_like(spec.fee_rate)), book)
        # Entry NAV=1/(1+5*.01)=20/21; exit pays another (20/21)*5*.01=1/21.
        np.testing.assert_allclose(result.risk.trading_fees, [[0., 1/21, 1/21]] * 2, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.risk.ending_equity, 19/21, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.prod(1 + result.returns, axis=1), 19/21, rtol=0, atol=1e-14)
        np.testing.assert_allclose(free.risk.ending_equity, 1., rtol=0, atol=1e-14)

    def test_trade_mark_basis_changes_post_trade_nav(self):
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                spec = flat_spec(days=2)
                trade = spec.trade_open.copy()
                trade[24] += sign
                result = self.run_book(replace(spec, trade_open=trade), target(spec, [[sign * 5.]]))
                # Adverse 1% execution basis at 5x: N=1-.05*N, with no trading fee.
                np.testing.assert_allclose(result.risk.ending_equity, 20/21, rtol=0, atol=1e-14)
                np.testing.assert_array_equal(result.risk.trading_fees, 0.)

    def test_collateral_is_redistributed_only_at_the_next_daily_rebalance(self):
        spec = flat_spec(days=3, assets=2)
        prices = {name: getattr(spec, name).copy() for name in
                  ('trade_open', 'mark_open', 'mark_high', 'mark_low', 'mark_close', 'funding_mark')}
        prices['mark_close'][47, 0] = prices['mark_high'][47, 0] = 110.
        for values in prices.values():
            values[48:, 0] = 110.
        prices['mark_close'][-1, 1] = prices['mark_high'][-1, 1] = 110.
        spec = replace(spec, **prices)
        book = np.full((1, *spec.shape), 2.5)
        result = self.run_book(spec, book)
        # Day one gains .025*10=.25. Day two reallocates q1=2.5*1.25/100=.03125.
        np.testing.assert_allclose(result.returns, [[0., .25, .25]], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.risk.ending_equity, 1.5625, rtol=0, atol=1e-14)

    def test_mark_touch_liquidates_long_and_short_at_the_threshold(self):
        for sign, numerator, denominator, adverse in ((1., 4., .04975, 80.), (-1., 6., .05025, 120.)):
            with self.subTest(sign=sign):
                spec = flat_spec(days=2)
                high, low = spec.mark_high.copy(), spec.mark_low.copy()
                if sign > 0:
                    low[25] = adverse
                else:
                    high[25] = adverse
                spec = replace(spec, mark_high=high, mark_low=low)
                result = self.run_book(spec, target(spec, [[sign * 5.]]))
                threshold = numerator / denominator
                liquidation_fee = .0005 * .05 * threshold
                remaining = (.005 - .0005) * .05 * threshold
                self.assert_event(result.risk, spec, 25, 8)
                np.testing.assert_allclose(result.risk.ending_equity, remaining, rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.risk.liquidation_fees.sum(1), liquidation_fee, rtol=0, atol=1e-14)
                np.testing.assert_array_equal(result.risk.isolated_loss_cap, 0.)
                np.testing.assert_array_equal(result.risk.bankrupt, False)

    def test_gap_bankruptcy_is_absorbing_and_does_not_invent_liquidation_fees(self):
        spec = flat_spec(days=4)
        prices = {name: getattr(spec, name).copy() for name in
                  ('trade_open', 'mark_open', 'mark_high', 'mark_low', 'mark_close', 'funding_mark')}
        for values in prices.values():
            values[25:] = 70.
        spec = replace(spec, **prices)
        book = np.full((1, *spec.shape), 5.)
        result = self.run_book(spec, book)
        self.assert_event(result.risk, spec, 25, 1)
        np.testing.assert_allclose(result.returns, [[0., -1., 0., 0.]], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.risk.isolated_loss_cap.sum(1), .5, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result.risk.liquidation_fees, 0.)
        np.testing.assert_array_equal(result.risk.ending_equity, 0.)
        np.testing.assert_array_equal(result.risk.bankrupt, True)

    def test_liquidation_fee_is_limited_to_the_remaining_bucket(self):
        spec = flat_spec(days=2)
        low = spec.mark_low.copy()
        low[25] = 80.
        spec = replace(spec, mark_low=low, liquidation_fee=np.full_like(low, .01))
        result = self.run_book(spec, target(spec, [[5.]]))
        self.assert_event(result.risk, spec, 25, 8)
        available = .005 * .05 * (4. / .04975)
        np.testing.assert_allclose(result.risk.liquidation_fees.sum(1), available, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result.risk.ending_equity, 0.)
        np.testing.assert_array_equal(result.risk.bankrupt, True)
        np.testing.assert_array_equal(result.risk.isolated_loss_cap, 0.)

    def test_isolated_loss_preserves_free_cash_and_other_asset_collateral(self):
        spec = flat_spec(days=2, assets=2)
        prices = {name: getattr(spec, name).copy() for name in
                  ('trade_open', 'mark_open', 'mark_high', 'mark_low', 'mark_close', 'funding_mark')}
        for values in prices.values():
            values[25:, 0] = 50.
        prices['mark_close'][-1, 1] = prices['mark_high'][-1, 1] = 110.
        spec = replace(spec, **prices)
        result = self.run_book(spec, target(spec, [[2.5, 0.], [2.5, 2.5]]))
        self.assert_event(result.risk, spec, 25, 1)
        # Asset zero loses its .5 bucket; excess .75 loss is isolated. Asset one retains q=.025.
        np.testing.assert_allclose(result.risk.isolated_loss_cap.sum(1), .75, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.risk.ending_equity, [.5, .75], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result.risk.bankrupt, False)
        np.testing.assert_array_equal(result.risk.liquidation_fees, 0.)

    def test_funding_liquidation_accounts_for_insufficient_isolated_balance(self):
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                spec = flat_spec(days=2)
                rates = spec.funding_rate.copy()
                rates[25] = sign * .4
                spec = replace(spec, funding_rate=rates)
                result = self.run_book(spec, target(spec, [[sign * 2.5]]))
                self.assert_event(result.risk, spec, 25, 2)
                np.testing.assert_allclose(result.risk.funding.sum(1), -1., rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.risk.isolated_loss_cap.sum(1), .5, rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.risk.ending_equity, .5, rtol=0, atol=1e-14)
                np.testing.assert_array_equal(result.risk.liquidation_fees, 0.)
                np.testing.assert_array_equal(result.risk.bankrupt, False)

    def test_entry_liquidation_uses_its_own_stage(self):
        spec = flat_spec(days=2, maintenance=.25)
        result = self.run_book(spec, target(spec, [[5.], [-5.]]))
        self.assert_event(result.risk, spec, 24, 4)
        np.testing.assert_allclose(result.risk.liquidation_fees.sum(1), .0025, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.risk.ending_equity, .9975, rtol=0, atol=1e-14)

    def test_future_events_and_decisions_do_not_rewrite_the_prefix(self):
        spec = flat_spec(days=4)
        book = np.full((1, *spec.shape), 2.5)
        original = self.run_book(spec, book)
        altered = book.copy()
        altered[:, 2:] *= -1       # the day-two close decision can first trade on day three
        same_events = self.run_book(spec, altered)
        np.testing.assert_array_equal(same_events.returns[:, :3], original.returns[:, :3])
        rates, marks = spec.funding_rate.copy(), spec.funding_mark.copy()
        rates[48:], marks[48:] = .4, 120.
        changed = self.run_book(replace(spec, funding_rate=rates, funding_mark=marks), altered)
        prefix = self.run_book(spec.rows(2), book[:, :2])
        for result in (changed, prefix):
            np.testing.assert_array_equal(result.returns[:, :2], original.returns[:, :2])
            for name in ('funding', 'trading_fees', 'liquidation_fees', 'isolated_loss_cap', 'liquidations'):
                np.testing.assert_array_equal(getattr(result.risk, name)[:, :2], getattr(original.risk, name)[:, :2])
            np.testing.assert_array_equal(result.risk.liquidation_events[:, :48], original.risk.liquidation_events[:, :48])
        self.assertGreater(changed.risk.liquidations.sum(), 0)

    def test_blend_keeps_signed_gross_budget_and_full_grid_risk(self):
        spec = flat_spec(days=3, assets=2, fee=.001)
        books = target(spec, [[5., 0.], [0., -5.]])
        base = np.zeros(spec.shape)
        base[0] = [0., -2.]
        owners, scales = np.array([0, 1, 0], dtype=np.int32), np.array([1., .5, .1])
        trials = cap_book(base[None] + scales[:, None, None] * books[owners], Allocation(5., True), np)
        expected = self.run_book(spec, trials, start=1)
        actual = self.executor.blend(self.xp.asarray(books), self.xp.asarray(base), self.xp.asarray(owners),
                                     self.xp.asarray(scales), self.executor.device_spec(spec), 1, 365., 'native', 2)
        actual = host_result(actual, self.to_host)
        np.testing.assert_allclose(actual.returns, expected.returns[:, 1:], rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual.statistics, expected.statistics, rtol=0, atol=1e-12)
        self.assertEqual(actual.risk.funding.shape, (3, 3))
        np.testing.assert_allclose(actual.risk.trading_fees, expected.risk.trading_fees, rtol=0, atol=1e-14)


class PerpetualCpuTests(HandCalculatedAccounts, unittest.TestCase):
    xp, to_host = np, staticmethod(np.asarray)

    @classmethod
    def setUpClass(cls):
        cls.executor = perpetual_cpu.prepare()

    @classmethod
    def tearDownClass(cls):
        cls.executor.close()


class AllocationTests(unittest.TestCase):
    def test_signal_budget_and_external_exposure_have_distinct_meanings(self):
        scores = np.array([[-2., .5, 3., np.nan], [1., -1., 1., -1.]])
        universe = np.array([[True, True, True, True], [True, True, False, False]])
        slots = np.maximum(universe.sum(1, keepdims=True), 1)
        allocation = Allocation(5., True)
        expected = np.array([[-.25, .125, .25, 0.], [.5, -.5, 0., 0.]])
        np.testing.assert_array_equal(signal_to_book(scores, universe, slots, allocation, np), expected)
        exposures = np.array([[3., -2.], [6., -4.]])
        np.testing.assert_allclose(cap_book(exposures, allocation, np), [[3., -2.], [3., -2.]], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

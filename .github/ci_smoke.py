# CI synthetic-data smoke, shared by every job in ci.yml (linux matrix / sanitizer / windows-msys2).
# No market data required: random-walk panels only. Exercises the C++ extensions through their real
# Python wrappers (INI config parsing + MarketProfile registry included) and guards the ABI contract.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))   # repo root (script lives in .github/)

import numpy as np

rng = np.random.default_rng(0)
T, S = 400, 12
close = np.ascontiguousarray(np.cumprod(1 + 0.01 * rng.standard_normal((T, S)), axis=0) * 100)
opn = np.ascontiguousarray(close * (1 + 0.001 * rng.standard_normal((T, S))))

# C operator library
from backtest.ops import cs_zscore_np
z = cs_zscore_np(np.ascontiguousarray(rng.standard_normal((T, S))))
assert z.shape == (T, S)

# all three portfolio backtest kernels via their Python wrappers
from rl.backtest_reward import (BacktestRewardConfig, EquityCostsConfig, AShareCostsConfig,
                                prepare_qnt_static, run_bt, run_equity_bt, run_ashare_bt)

high = np.ascontiguousarray(close * 1.01)
low = np.ascontiguousarray(close * 0.99)
qs = prepare_qnt_static(opn, high, low, close)
wq = np.ascontiguousarray(np.abs(rng.standard_normal((T, S))))
r = run_bt(wq, qs, 20, BacktestRewardConfig())
print('crypto ', {k: round(v, 4) for k, v in r.items()})

sig = np.ascontiguousarray(rng.standard_normal((T, S)))
w = sig - sig.mean(1, keepdims=True)
w /= np.abs(w).sum(1, keepdims=True)
r = run_equity_bt(np.ascontiguousarray(w), close, opn, EquityCostsConfig())
print('equity ', {k: round(v, 4) for k, v in r.items()})

wl = np.full((T, S), 1.0 / S)
r = run_ashare_bt(wl, close, opn, AShareCostsConfig())
print('ashare ', {k: round(v, 4) for k, v in r.items()})

# bars_per_year / ppy is a required positional arg on every kernel (no hidden default)
from backtest import _bt
zero3 = np.zeros((10, 2))
one3 = np.ones((10, 2))
ul3 = np.ones((10, 2), dtype=np.uint8)
for name, args in (
    ('portfolio_bt_equity', (zero3, one3, one3, 1e5, 0, 0.0005, 0.0)),
    ('portfolio_bt_ashare', (zero3, one3, one3, 1e5, 0, 0.00025, 0.0005, 0.00001)),
    ('portfolio_bt_qnt',    (zero3, one3, one3, one3, one3, one3, ul3, 0)),
):
    try:
        getattr(_bt, name)(*args)
    except TypeError:
        pass
    else:
        raise SystemExit(f'{name}: bars_per_year must be a required argument')
print('smoke OK')

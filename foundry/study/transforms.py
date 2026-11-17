"""Pure source transforms shared by target and account replay orchestration."""
import hashlib
from collections.abc import Sequence
from types import ModuleType

from foundry.types import FloatArray


DELAY_LAG = (365, 1095)
INVALID = 'G1 数值无效:含无穷值或全为缺失'
EXPOSURE = 'G1 gross exposure exceeds the configured account budget'


def delay_lag(key: str) -> int:
    """Deterministic forward-only noise control; the sample end never wraps."""
    lo, hi = DELAY_LAG
    return lo + int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (hi - lo + 1)


def delay_shift(values: FloatArray, keys: Sequence[str], xp: ModuleType, fill: float) -> FloatArray:
    """Shift (B,T,S) values by causal identities without wrapping or state."""
    lag = xp.asarray([delay_lag(key) for key in keys])
    at = xp.arange(values.shape[1])[None, :] - lag[:, None]
    return xp.where((at >= 0)[:, :, None],
                    values[xp.arange(len(keys))[:, None], xp.maximum(at, 0)], fill)

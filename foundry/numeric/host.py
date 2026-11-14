"""Host implementation of the common return-moments contract."""
import numpy as np

from foundry.contracts import ReturnMoments
from foundry.core import metrics


def measure(returns, weights, ppy):
    return ReturnMoments(metrics.sharpe(returns, ppy, np),
                         metrics.cagr(returns, ppy, np),
                         metrics.weighted_sharpe(returns, weights, ppy, np),
                         metrics.recent_growth(returns, weights, ppy, np))

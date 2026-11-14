"""Loaded perpetual observations, before market rules compile to execution arrays."""
from dataclasses import dataclass
from foundry.contracts import MarketData


@dataclass(frozen=True)
class PerpetualData:
    daily: MarketData
    event_times_ms: object
    day_index: object
    trade_open: object
    mark_open: object
    mark_high: object
    mark_low: object
    mark_close: object
    funding_rate: object
    funding_mark: object

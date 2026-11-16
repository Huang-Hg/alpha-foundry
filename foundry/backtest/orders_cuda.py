"""Explicitly prepared CUDA streaming order accounts; no host fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from foundry.backtest.orders import OrderRuntime as HostOrderRuntime, PreparedData
from foundry.config import Account, Execution
from foundry.numeric.cuda_source import paired_source
from foundry.types import BarExecution

if TYPE_CHECKING:
    from cupy import RawModule


@dataclass(frozen=True)
class PreparedBackend:
    module: RawModule


def prepare(array_module: ModuleType) -> PreparedBackend:
    """Compile both declared widths during explicit resource initialization."""
    root = Path(__file__).parent
    source = paired_source() + '\n' + (root/'src'/'orders.hpp').read_text(encoding='utf-8')
    source += '\n' + (root/'kernels'/'orders.cu').read_text(encoding='utf-8')
    module = array_module.RawModule(code=source, options=('--std=c++17', '--fmad=false'))
    module.get_function('orders_paired')
    module.get_function('orders_mixed')
    return PreparedBackend(module)


class OrderRuntime(HostOrderRuntime):
    backend = 'cuda'

    def __init__(self, spec: BarExecution, account: Account, execution: Execution,
                 batch: int = 1, precision: str = 'paired', *, array_module: ModuleType,
                 prepared: PreparedBackend, prepared_data: PreparedData, start_event: int = 0):
        self._initialize(spec, account, execution, batch, precision, array_module, prepared_data, start_event)
        self.kernel = prepared.module.get_function('orders_'+precision)
        # A paired scalar occupies eight bytes; mixed scratch uses its prefix.
        self._scratch = array_module.empty((batch, self.assets), dtype=array_module.float64)

    def _step(self, event: int, measured: bool) -> None:
        self.kernel(((self.batch+31)//32,), (32,),
            (self._panels[event], self._rules[event], self._assets, self._cash,
             self._meta, self._oi, self._ov, self._ci, self._cv, self._ai, self._av, self._count, self._scratch,
             np.int64(self.batch), np.int64(self.assets), np.int64(self.capacity), np.int64(self.command_capacity),
             np.int64(self.feedback_capacity), np.int64(self.spec.schedule.event_ns[event]),
             np.int32(self.account.margin == 'isolated'),
             np.int32(self.execution.bar_path == 'open_high_low_close'),
             np.int32(measured),
             np.float64(self.account.leverage), np.float64(self.account.gross_limit), np.float64(self.execution.participation)))

    def close(self) -> None:
        if not self.closed:
            super().close()
            del self._scratch, self.kernel

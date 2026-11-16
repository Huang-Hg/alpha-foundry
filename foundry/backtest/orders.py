"""Bounded streaming order accounts; native resources are supplied at explicit open."""
from __future__ import annotations

from dataclasses import dataclass, fields
from types import ModuleType

import numpy as np

from foundry.config import Account, Execution
from foundry.numeric.semantics import ORDER_SEMANTIC_VERSION
from foundry.types import (
    AccountSnapshot, Array, BarExecution, ExecutionFeedback, ExecutionRiskBatch, FloatArray, OrderCommands, OrderSnapshot,
)


PANEL_FIELDS = ('trade_open', 'trade_high', 'trade_low', 'trade_close', 'volume',
    'mark_open', 'mark_high', 'mark_low', 'mark_close', 'fee_buy', 'fee_sell',
    'funding_rate', 'funding_mark', 'maintenance_rate', 'liquidation_fee', 'qty_step', 'price_tick')
RULE_FIELDS = ('buy_allowed', 'sell_allowed', 'quote_valid')
COMMAND_INTS = ('action', 'owner', 'id', 'asset', 'side', 'type', 'valid_until', 'reduce_only')
COMMAND_VALUES = ('quantity', 'price', 'reservation_price')


@dataclass(frozen=True)
class PreparedData:
    """Shared immutable market buffers owned by the explicit study runtime."""
    spec: BarExecution
    panels: FloatArray
    rules: Array
    event_ns: Array
    xp: ModuleType


def prepare_data(spec: BarExecution, array_module: ModuleType = np) -> PreparedData:
    """Pack/transfer shared market inputs once during explicit resource open."""
    panels = array_module.asarray(np.stack([getattr(spec, name) for name in PANEL_FIELDS], axis=1))
    rules = array_module.asarray(np.stack([getattr(spec, name) for name in RULE_FIELDS], axis=1).astype(np.uint8))
    if array_module is np:
        panels.flags.writeable = False
        rules.flags.writeable = False
    event_ns = array_module.asarray(spec.schedule.event_ns)
    return PreparedData(spec, panels, rules, event_ns, array_module)


class OrderRuntime:
    """Own O(B*(M*S+Q+K+F)) mutable state and O(B*V) measured output.

    Each advance processes one complete base bar. Commands originate at its
    compiled decision cutoff; callbacks and causal feature windows are owned by
    study. An optional user sink consumes each returned bounded feedback batch.
    """
    backend = 'cpu'

    def __init__(self, spec: BarExecution, account: Account, execution: Execution,
                 batch: int = 1, precision: str = 'paired', *, native: ModuleType,
                 prepared_data: PreparedData, start_event: int = 0, owner_count: int = 1,
                 initial_cash: FloatArray | None = None):
        self.native = native
        self._initialize(spec, account, execution, batch, precision, np, prepared_data, start_event, owner_count, initial_cash)

    def _initialize(self, spec: BarExecution, account: Account, execution: Execution,
                    batch: int, precision: str, xp: ModuleType, prepared_data: PreparedData,
                    start_event: int, owner_count: int, initial_cash: FloatArray | None) -> None:
        if not isinstance(spec, BarExecution):
            raise TypeError('Order execution requires a compiled BarExecution')
        if spec.market not in ('crypto_spot', 'crypto_perpetual'):
            raise ValueError('Equity program orders require compiled corporate-action and settlement capabilities')
        if (spec.market == 'crypto_perpetual') != (account.margin == 'isolated'):
            raise ValueError('Bar market and cash/isolated account model must agree')
        if type(batch) is not int or batch < 1:
            raise ValueError('Order account batch must be a positive integer')
        if precision not in ('paired', 'mixed'):
            raise ValueError('Order execution precision must be paired or mixed')
        if prepared_data.spec is not spec or prepared_data.xp is not xp:
            raise ValueError('Prepared order market data must belong to this spec and backend')
        if type(start_event) is not int or not 0 <= start_event < len(spec.schedule.event_ns):
            raise ValueError('Order runtime start event must lie inside the compiled event clock')
        if type(owner_count) is not int or owner_count < 1:
            raise ValueError('Order account owner_count must be a positive integer')
        capital = xp.full(batch, account.initial_cash, dtype=xp.float64) if initial_cash is None else xp.asarray(initial_cash, dtype=xp.float64)
        if capital.shape != (batch,) or bool(xp.any(~xp.isfinite(capital) | (capital < 0))):
            raise ValueError('Private account capital requires finite nonnegative quote amounts (B,)')
        self.owner_count = owner_count
        self.spec, self.account, self.execution = spec, account, execution
        self.xp, self.batch, self.precision = xp, batch, precision
        self.assets = spec.trade_open.shape[1]
        self.capacity, self.command_capacity = execution.order_capacity, execution.command_capacity
        self.feedback_capacity = (self.command_capacity + owner_count*self.assets + 4*self.capacity
                                  + (owner_count+1)*(self.capacity+self.assets))
        self.cursor, self.closed = start_event, False
        self._panels, self._rules = prepared_data.panels, prepared_data.rules
        self._event_times = prepared_data.event_ns
        self._assets = xp.zeros((batch, self.assets, 4), dtype=xp.float64)
        self._assets[:, :, 2] = self._panels[start_event, 5]
        self._cash = xp.zeros((batch, 14), dtype=xp.float64)
        self._cash[:, :3] = capital[:, None]
        self._meta = xp.zeros((batch, 5), dtype=xp.int64)
        self._meta[:, 3] = capital <= 0
        self._meta[:, 4] = -1
        self._owners = xp.zeros((batch, owner_count, self.assets), dtype=xp.float64)
        self._highwater = xp.zeros((batch, owner_count), dtype=xp.int64)
        self._targets = xp.zeros_like(self._owners)
        self._updates = xp.zeros(self._owners.shape, dtype=xp.uint8)
        self._units = xp.zeros(owner_count, dtype=xp.int64)
        self._coefficients = xp.ones((batch, owner_count), dtype=xp.float64)
        self._pending_targets = xp.zeros_like(self._owners)
        self._target_dirty = xp.zeros(self._owners.shape, dtype=xp.uint8)
        self._activity = xp.zeros((batch, 6), dtype=xp.float64)
        self._digest = xp.zeros((batch, 26), dtype=xp.uint32)
        self._digest[:, :8] = xp.asarray((0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19), dtype=xp.uint32)
        self._oi = xp.zeros((batch, self.capacity, 9), dtype=xp.int64)
        self._ov = xp.zeros((batch, self.capacity, 6), dtype=xp.float64)
        self._ci = xp.zeros((batch, self.command_capacity, 8), dtype=xp.int64)
        self._cv = xp.zeros((batch, self.command_capacity, 3), dtype=xp.float64)
        self._ai = xp.zeros((batch, self.feedback_capacity, 8), dtype=xp.int64)
        self._av = xp.zeros((batch, self.feedback_capacity, 4), dtype=xp.float64)
        self._count = xp.zeros(batch, dtype=xp.int64)
        shape = (batch, spec.schedule.measurement_count)
        self._returns = xp.zeros(shape, dtype=xp.float64)
        self._equity = xp.broadcast_to(capital[:, None], shape).copy()
        self._active = xp.zeros(shape, dtype=xp.bool_)
        self._risk_flows = xp.zeros((batch, spec.schedule.measurement_count, 4), dtype=xp.float64)
        self._risk_counts = xp.zeros(shape, dtype=xp.int64)
        self._risk_previous = xp.zeros((batch, 5), dtype=xp.float64)

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError('Order runtime is closed')

    def _detached(self, value: FloatArray) -> FloatArray:
        owned = value.copy()
        if self.backend == 'cpu':
            owned.flags.writeable = False
        return owned

    def snapshot(self) -> AccountSnapshot:
        """Return detached actual balances and active orders, never ledger aliases."""
        self._ensure_open()
        xp = self.xp
        active = self._oi[:, :, 0] != 0
        status = xp.where(active, xp.where(self._ov[:, :, 0] < self._ov[:, :, 4], 2, 1), 0)
        orders = OrderSnapshot(
            self._detached(active.astype(xp.int64)), self._detached(self._oi[:, :, 1]),
            self._detached(self._oi[:, :, 0]), self._detached(self._oi[:, :, 2]),
            self._detached(self._oi[:, :, 3]), self._detached(self._ov[:, :, 4]),
            self._detached(self._oi[:, :, 4]), self._detached(self._ov[:, :, 1]),
            self._detached(self._ov[:, :, 2]), self._detached(self._oi[:, :, 5]),
            self._detached(self._oi[:, :, 7].astype(xp.bool_)), self._detached(self._ov[:, :, 0]),
            self._detached(self._ov[:, :, 3]), self._detached(status))
        return AccountSnapshot(self._detached(self._cash[:, 0]), self._detached(self._cash[:, 1]),
            self._detached(self._cash[:, 3]), self._detached(self._assets[:, :, 0]),
            self._detached(self._assets[:, :, 1]), self._detached(self._assets[:, :, 3]), orders)

    def _commands(self, commands: OrderCommands) -> None:
        if not isinstance(commands, OrderCommands):
            raise TypeError('Programs must return OrderCommands')
        xp = self.xp
        if self.backend == 'cpu' and not isinstance(commands.action, np.ndarray):
            raise TypeError('CPU order commands must be host NumPy arrays')
        if self.backend == 'cuda' and not hasattr(commands.action, '__cuda_array_interface__'):
            raise TypeError('CUDA order commands must remain on the declared device')
        shape = commands.action.shape
        if len(shape) != 2 or shape[0] != self.batch or shape[1] > self.command_capacity:
            raise ValueError('Command columns require bounded (B,K) coordinates')
        for item in fields(commands):
            value = getattr(commands, item.name)
            if self.backend == 'cpu':
                if not isinstance(value, np.ndarray):
                    raise TypeError('CPU order commands must be host NumPy arrays')
            elif not hasattr(value, '__cuda_array_interface__'):
                raise TypeError('CUDA order commands must remain on the declared device')
            if value.shape != shape:
                raise ValueError('Every command column must share the same (B,K) coordinates')
            if item.name == 'reduce_only':
                valid_dtype = value.dtype == np.dtype('bool')
            elif item.name in COMMAND_INTS:
                valid_dtype = np.issubdtype(value.dtype, np.signedinteger)
            else:
                valid_dtype = np.issubdtype(value.dtype, np.floating)
            if not valid_dtype:
                raise TypeError('Command columns require signed integer, floating or bool declared dtypes')
        active = commands.action != 0
        new = active & (commands.action != 2)
        invalid = (commands.action < 0) | (commands.action > 3)
        invalid |= active & ((commands.owner < 0) | (commands.owner >= self.owner_count) | (commands.id <= 0))
        invalid |= new & ((commands.asset < 0) | (commands.asset >= self.assets)
            | ((commands.side != 1) & (commands.side != -1))
            | (commands.type < 1) | (commands.type > 3)
            | ~xp.isfinite(commands.quantity) | (commands.quantity <= 0)
            | ~xp.isfinite(commands.reservation_price) | (commands.reservation_price <= 0)
            | ~xp.isfinite(commands.price) | (commands.price < 0)
            | ((commands.type != 1) & (commands.price <= 0))
            | (commands.valid_until < -1))
        if bool(xp.any(invalid)):
            raise ValueError('Order command values violate the declared input contract')
        expiry = commands.valid_until
        positions = xp.searchsorted(self._event_times, expiry)
        positions = xp.minimum(positions, len(self.spec.schedule.event_ns)-1)
        if bool(xp.any(new & (expiry != -1) & (self._event_times[positions] != expiry))):
            raise ValueError('Order expiry must be -1 or a compiled execution-bar open timestamp')
        self._ci.fill(0)
        self._cv.fill(0)
        size = shape[1]
        for index, name in enumerate(COMMAND_INTS):
            self._ci[:, :size, index] = getattr(commands, name)
        for index, name in enumerate(COMMAND_VALUES):
            self._cv[:, :size, index] = getattr(commands, name)
        # Cancel payload fields are irrelevant; normalize them before the raw
        # array boundary so no unused user coordinate can become an index.
        cancel = self._ci[:, :, 0] == 2
        self._ci[:, :, 3] = xp.where(cancel, 0, self._ci[:, :, 3])
        self._ci[:, :, 4] = xp.where(cancel, 1, self._ci[:, :, 4])
        self._ci[:, :, 5] = xp.where(cancel, 1, self._ci[:, :, 5])

    def _step(self, event: int, measured: bool) -> None:
        inputs = (self._panels[event], self._rules[event], self._assets, self._cash,
                  self._meta, self._oi, self._ov, self._ci, self._cv, self._ai, self._av, self._count,
                  self._owners, self._highwater, self._targets, self._updates, self._units, self._coefficients,
                  self._pending_targets, self._target_dirty, self._activity, self._digest)
        self.native.step(inputs, int(self.spec.schedule.event_ns[event]),
            int(self.account.margin == 'isolated'), int(self.execution.bar_path == 'open_high_low_close'),
            int(measured), self.account.leverage, self.account.gross_limit, self.execution.participation, self.precision)

    def configure_targets(self, units: tuple[str, ...], coefficients: FloatArray) -> None:
        """Bind immutable member units and per-trial intent scales before replay."""
        self._ensure_open()
        if len(units) != self.owner_count or any(unit not in ('equity_fraction', 'quantity', 'notional') for unit in units):
            raise ValueError('Target units must name one supported unit per owner')
        values = self.xp.asarray(coefficients, dtype=self.xp.float64)
        if values.shape != (self.batch, self.owner_count) or bool(self.xp.any(~self.xp.isfinite(values) | (values < 0))):
            raise ValueError('Intent coefficients require finite nonnegative (B,M) values')
        self._units[:] = self.xp.asarray([('equity_fraction', 'quantity', 'notional').index(unit) for unit in units], dtype=self.xp.int64)
        self._coefficients[:] = values

    def owner_quantities(self, owner: int | None = None) -> FloatArray:
        """Detach bounded signed owner contributions (B,M,S), whose sum is net quantity."""
        self._ensure_open()
        if owner is not None and (type(owner) is not int or not 0 <= owner < self.owner_count):
            raise ValueError('Owner index must lie inside the declared owner namespace')
        return self._detached(self._owners if owner is None else self._owners[:, owner])

    def advance(self, event: int, commands: OrderCommands, *, targets: FloatArray | None = None,
                update_mask: Array | None = None) -> ExecutionFeedback:
        """Advance exactly the next base event and detach its bounded audit."""
        self._ensure_open()
        if type(event) is not int or event != self.cursor or event >= len(self.spec.schedule.event_ns):
            raise ValueError('Order events must advance consecutively within the compiled schedule')
        self._commands(commands)
        self._updates.fill(0)
        if targets is not None:
            values = self.xp.asarray(targets)
            if values.shape != self._targets.shape or values.dtype not in (self.xp.dtype('float32'), self.xp.dtype('float64')):
                raise ValueError('Target event values require floating (B,M,S) coordinates')
            if bool(self.xp.any(~self.xp.isfinite(values))) or (self.account.margin == 'cash' and bool(self.xp.any(values < 0))):
                raise ValueError('Targets must be finite and cash targets cannot be negative')
            mask = self.xp.ones(values.shape, dtype=self.xp.bool_) if update_mask is None else self.xp.asarray(update_mask)
            if mask.shape != values.shape or mask.dtype != self.xp.dtype('bool'):
                raise ValueError('Target updates require bool (B,M,S) coordinates')
            self._targets[:] = values
            self._updates[:] = mask
        elif update_mask is not None:
            raise ValueError('Target update masks require event target values')
        self._ai.fill(0)
        self._av.fill(0)
        measurement = int(self.spec.schedule.measurement_index[event])
        measured = event+1 == len(self.spec.schedule.event_ns) or self.spec.schedule.measurement_index[event+1] != measurement
        self._step(event, measured)
        if measured:
            self._returns[:, measurement] = self._cash[:, 10]
            self._equity[:, measurement] = self._cash[:, 11]
            self._active[:, measurement] = self.xp.any(self._assets[:, :, 0] != 0, axis=1)
            cumulative = self.xp.stack((self._cash[:, 6], self._cash[:, 5], self._cash[:, 7],
                                        self._cash[:, 8], self._meta[:, 2]), axis=1)
            delta = cumulative - self._risk_previous
            self._risk_flows[:, measurement] = delta[:, :4]
            self._risk_counts[:, measurement] = delta[:, 4].astype(self.xp.int64)
            self._risk_previous[:] = cumulative
        self.cursor += 1
        return ExecutionFeedback(self.snapshot(), self._detached(self._count),
            *[self._detached(self._ai[:, :, index]) for index in (1, 2, 3, 4, 5, 6, 7, 0)],
            *[self._detached(self._av[:, :, index]) for index in range(4)],
            event_ns=int(self.spec.schedule.event_ns[event]), available_ns=int(self.spec.event_close_ns[event]))

    def returns(self) -> FloatArray:
        self._ensure_open()
        return self._detached(self._returns)

    def equity(self) -> FloatArray:
        self._ensure_open()
        return self._detached(self._equity)

    def active(self) -> Array:
        self._ensure_open()
        return self._detached(self._active)

    def bar_activity(self) -> FloatArray:
        """Detach current bar traded notional and opening equity in quote units."""
        self._ensure_open()
        return self._detached(self._cash[:, 12:14])

    def activity(self) -> FloatArray:
        """Running [turnover,gross sum,net sum,max asset,fill count,valuations]."""
        self._ensure_open()
        return self._detached(self._activity)

    def risk(self, normalization: float) -> ExecutionRiskBatch:
        """Measured flows in initial-NAV units; event audits remain streaming only."""
        self._ensure_open()
        xp = self.xp
        flows = self._risk_flows / normalization
        return ExecutionRiskBatch(*(self._detached(flows[:, :, i]) for i in range(4)),
            self._detached(self._risk_counts), xp.empty((self.batch, 0, self.assets), dtype=xp.uint8),
            self._detached(self._meta[:, 4]), self._detached(self._meta[:, 3].astype(xp.bool_)),
            self._detached(self._cash[:, 1] / normalization), self._detached(self._cash[:, 9]),
            ORDER_SEMANTIC_VERSION)

    def _finish_digest(self, state: Array) -> None:
        self.native.digest(state)

    def behavior_digests(self) -> tuple[str, ...]:
        """Finalize detached SHA-256 states without closing or modifying the account."""
        self._ensure_open()
        state = self._digest.copy()
        self._finish_digest(state)
        host = np.asarray(state[:, :8]) if self.backend == 'cpu' else self.xp.asnumpy(state[:, :8])
        return tuple(''.join(f'{int(word):08x}' for word in row) for row in host)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            for name in ('_panels', '_rules', '_event_times', '_assets', '_cash', '_meta', '_oi', '_ov',
                         '_ci', '_cv', '_ai', '_av', '_count', '_returns', '_equity', '_owners', '_highwater',
                         '_targets', '_updates', '_units', '_coefficients', '_pending_targets', '_target_dirty',
                         '_activity', '_digest', '_active', '_risk_flows', '_risk_counts', '_risk_previous'):
                delattr(self, name)

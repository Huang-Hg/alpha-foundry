"""Typed grammar metadata and the independent host uniform baseline."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from foundry.search.proposer.grammar import grammar
from foundry.types import HostArray


@dataclass(frozen=True)
class Action:
    name: str
    output: int
    cost: int
    children: tuple[int, ...]

    @property
    def temporal(self) -> bool:
        return self.name.startswith('ts_')


@dataclass(frozen=True)
class State:
    sequence: tuple[int, ...] = ()
    slots: tuple[int, ...] = (0,)
    nesting: tuple[int, ...] = (0,)
    cost: int = 0


class Sampler:
    """Bounded current-batch state; no market tensors or history cache."""

    def __init__(self, fields: Sequence[str], seed: int, max_tokens: int, max_cost: int,
                 max_ts_depth: int, window_caps: tuple[int, int], context_size: int) -> None:
        if not 3 <= max_tokens <= 64 or not 3 <= max_cost <= 100 or not 0 <= max_ts_depth <= 64:
            raise ValueError('invalid grammar budget')
        if context_size <= 0:
            raise ValueError('invalid context dimension')
        self.spec = grammar(list(fields), window_caps)
        self.actions = tuple(Action(row[0], int(row[1]), int(row[2]), tuple(map(int, row[4:])))
                             for row in (line.split() for line in self.spec.splitlines()))
        for slot in range(3):
            if not any(a.output == slot and not a.children for a in self.actions):
                raise ValueError('grammar lacks terminal for a slot')
        self.max_tokens, self.max_cost, self.max_ts_depth = max_tokens, max_cost, max_ts_depth
        self.context_size = context_size
        self.rng = np.random.default_rng(seed)
        self.context = np.zeros(context_size, dtype=np.float32)
        self.outputs = np.array([a.output for a in self.actions])
        self.costs = np.array([a.cost for a in self.actions])
        self.arities = np.array([len(a.children) for a in self.actions])
        self.child_costs = np.array([sum(c in (0, 1, 3) for c in a.children) for a in self.actions])
        self.temporal = np.array([a.temporal for a in self.actions])
        self.nonzero = np.array([a.output == 1 and float(a.name) != 0 for a in self.actions])
        self.epoch, self.ordinary = 0, 0
        self.samples: list[State] = []

    def legal(self, state: State) -> HostArray:
        slot = state.slots[-1]
        need = state.cost + self.costs + sum(c in (0, 1, 3) for c in state.slots[:-1]) + self.child_costs
        return (((self.outputs == slot) | ((slot == 3) & self.nonzero)) &
                (~self.temporal | (state.nesting[-1] < self.max_ts_depth)) &
                (need <= self.max_cost) &
                (len(state.sequence) + len(state.slots) + self.arities <= self.max_tokens))

    def step(self, state: State, index: int) -> State:
        action = self.actions[index]
        children = tuple(reversed(action.children))
        return State(state.sequence + (index,), state.slots[:-1] + children,
                     state.nesting[:-1] + (state.nesting[-1] + action.temporal,) * len(children),
                     state.cost + action.cost)

    def choose(self, probabilities: HostArray) -> HostArray:
        cumulative = np.cumsum(probabilities, axis=1)
        draws = self.rng.random((len(probabilities), 1)) * cumulative[:, -1:]
        return (draws > cumulative).sum(axis=1)

    def sample(self, count: int, context: Sequence[float]) -> None:
        if not 0 < count <= 8192:
            raise ValueError('invalid batch')
        values = np.asarray(context, dtype=np.float32)
        if values.shape != (self.context_size,) or not np.isfinite(values).all():
            raise ValueError(f'context needs {self.context_size} finite values')
        self.context = values
        self.epoch += 1
        self.ordinary = count
        self.samples = [State()] * count
        while True:
            active = [i for i, state in enumerate(self.samples) if state.slots]
            if not active:
                return
            choices = self.choose(np.stack([self.legal(self.samples[i]) for i in active]))
            for index, action in zip(active, choices):
                self.samples[index] = self.step(self.samples[index], int(action))

    def expression(self, sequence: tuple[int, ...]) -> str:
        def render(position: int) -> tuple[str, int]:
            action = self.actions[sequence[position]]
            position += 1
            children = []
            for _ in action.children:
                child, position = render(position)
                children.append(child)
            return action.name + ('(' + ','.join(children) + ')' if children else ''), position
        return render(0)[0]

    def expressions(self) -> tuple[str, ...]:
        return tuple(self.expression(state.sequence) for state in self.samples)

"""Typed prefix trajectories and paired common-support probes on host metadata."""
from __future__ import annotations

from collections.abc import Callable, Sequence
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


@dataclass(frozen=True)
class Probe:
    state: State
    actions: tuple[int, ...]
    ids: tuple[int, ...]


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
        self.probes: list[Probe] = []

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

    def features(self, state: State) -> HostArray:
        c = self.context_size
        x = np.zeros(c + 64, dtype=np.float32)
        x[0], x[1:c + 1] = 1, self.context
        x[c + 1:c + 4] = (len(state.sequence) / self.max_tokens,
                          state.cost / self.max_cost, len(state.slots) / self.max_tokens)
        x[c + 4 + state.slots[-1]] = 1
        scale = 1 / np.sqrt(max(1, len(state.sequence)))
        for position, action in enumerate(state.sequence):
            hashed = (((action + 1) * 0x9e3779b97f4a7c15) ^
                      ((position + 1) * 0xbf58476d1ce4e5b9)) & ((1 << 64) - 1)
            x[c + 10 + hashed % 54] += scale if hashed >> 63 else -scale
        return x

    def encode(self, states: Sequence[State]) -> tuple[HostArray, HostArray]:
        return np.stack([self.features(s) for s in states]), np.stack([self.legal(s) for s in states])

    def choose(self, probabilities: HostArray) -> HostArray:
        cumulative = np.cumsum(probabilities, axis=1)
        draws = self.rng.random((len(probabilities), 1)) * cumulative[:, -1:]
        return (draws > cumulative).sum(axis=1)

    def rollout(self, states: list[State], probabilities: Callable[[Sequence[State]], HostArray],
                *, paired: bool = False) -> list[State]:
        while True:
            active = [i for i, state in enumerate(states) if state.slots]
            if not active:
                return states
            values = probabilities([states[i] for i in active])
            if paired:
                common = np.cbrt(np.prod(values.reshape(-1, 3, values.shape[-1]), axis=1))
                if np.any(common.sum(axis=1) == 0):
                    raise RuntimeError('empty common support')
                chosen = np.repeat(self.choose(common), 3)
            else:
                chosen = self.choose(values)
            for index, action in zip(active, chosen):
                states[index] = self.step(states[index], int(action))

    def sample(self, count: int, probes: int, context: Sequence[float],
               probabilities: Callable[[Sequence[State]], HostArray], *, uniform: bool = False) -> None:
        if not 0 < count <= 8192 or not 0 <= probes <= count:
            raise ValueError('invalid batch')
        values = np.asarray(context, dtype=np.float32)
        if values.shape != (self.context_size,) or not np.isfinite(values).all():
            raise ValueError(f'context needs {self.context_size} finite values')
        self.context = values
        self.epoch += 1
        self.ordinary = count
        self.probes = []
        self.samples = self.rollout([State()] * count, probabilities)
        if uniform:
            return
        for _ in range(probes * 24):
            if len(self.probes) == probes:
                break
            source = self.samples[int(self.rng.integers(count))]
            position = int(self.rng.integers(len(source.sequence)))
            base = State()
            for action in source.sequence[:position]:
                base = self.step(base, action)
            first_index = source.sequence[position]
            first = self.actions[first_index]
            p = probabilities([base])[0]
            siblings = [i for i, a in enumerate(self.actions) if i != first_index and p[i] > 1e-4 and
                        (a.children, a.cost, a.output) == (first.children, first.cost, first.output)]
            if len(siblings) < 2:
                continue
            self.rng.shuffle(siblings)
            actions = (first_index, *siblings[:2])
            # Generate at most 40 suffix attempts, four at a time, with shared draws per branch triple.
            rows: list[State] = []
            suffixes: set[tuple[int, ...]] = set()
            for _ in range(10):
                branches = self.rollout([self.step(base, a) for _ in range(4) for a in actions],
                                        probabilities, paired=True)
                for k in range(4):
                    suffix = branches[3 * k].sequence[position + 1:]
                    if suffix not in suffixes:
                        suffixes.add(suffix)
                        rows.extend(branches[3 * k:3 * k + 3])
                    if len(suffixes) == 4:
                        break
                if len(suffixes) == 4:
                    break
            if len(suffixes) != 4:
                continue
            ids = tuple(range(len(self.samples), len(self.samples) + 12))
            self.samples.extend(rows)
            self.probes.append(Probe(base, actions, ids))

    def expressions(self) -> tuple[str, ...]:
        def render(sequence: tuple[int, ...], position: int) -> tuple[str, int]:
            action = self.actions[sequence[position]]
            position += 1
            children = []
            for _ in action.children:
                child, position = render(sequence, position)
                children.append(child)
            return action.name + ('(' + ','.join(children) + ')' if children else ''), position
        return tuple(render(s.sequence, 0)[0] for s in self.samples)

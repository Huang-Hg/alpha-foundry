"""Device-resident prefix sampling with bounded paired-probe workspaces."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from foundry.search.proposer.sampling import Sampler


@dataclass(frozen=True)
class Trajectories:
    """Current-batch device tensors: N prefix rows, B trajectories, P probes, A actions."""

    features: Tensor  # (N,F), float32
    legal: Tensor  # (N,A), bool
    actions: Tensor  # (N,), int64
    owners: Tensor  # (B,L), int64 prefix row indices with an appended zero sentinel
    context: Tensor  # (C+1,), float32
    probe_features: Tensor  # (P,F), float32
    probe_legal: Tensor  # (P,A), bool
    probe_actions: Tensor  # (P,3), int64
    probe_ids: Tensor  # (P,4,3), int64


@dataclass(frozen=True)
class Prefixes:
    tokens: Tensor  # (B,L), int64, -1 padding
    slots: Tensor  # (B,L+4), int64, stack storage
    nesting: Tensor  # (B,L+4), int64
    depth: Tensor  # (B,), int64
    length: Tensor  # (B,), int64
    cost: Tensor  # (B,), int64
    hashed: Tensor  # (B,54), float32, signed prefix counts

    def select(self, indices: Tensor) -> Prefixes:
        return Prefixes(*(value.index_select(0, indices) for value in (
            self.tokens, self.slots, self.nesting, self.depth, self.length, self.cost, self.hashed)))


@dataclass(frozen=True)
class Drawn:
    trace: Trajectories
    tokens: Tensor  # (B+12P,L), int64; only final expressions cross to the host
    probe_lengths: Tensor  # (P,), int64


class DeviceSampler:
    """One grammar and RNG on the selected device; storage scales with the current batch."""

    def __init__(self, grammar: Sampler, device: torch.device, seed: int) -> None:
        self.device = device
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.length, self.cost_limit, self.nest_limit = grammar.max_tokens, grammar.max_cost, grammar.max_ts_depth
        self.context_size = grammar.context_size
        self.outputs = torch.as_tensor(grammar.outputs, device=device)
        self.costs = torch.as_tensor(grammar.costs, device=device)
        self.arities = torch.as_tensor(grammar.arities, device=device)
        self.child_costs = torch.as_tensor(grammar.child_costs, device=device)
        self.temporal = torch.as_tensor(grammar.temporal, device=device)
        self.nonzero = torch.as_tensor(grammar.nonzero, device=device)
        self.children = torch.tensor([list(reversed(a.children)) + [0] * (4 - len(a.children))
                                      for a in grammar.actions], dtype=torch.int64, device=device)
        keys = [(a.output, a.cost, a.children) for a in grammar.actions]
        self.groups = torch.tensor([keys.index(key) for key in keys], device=device)
        hashes = [[(((a + 1) * 0x9e3779b97f4a7c15) ^ ((p + 1) * 0xbf58476d1ce4e5b9)) &
                   ((1 << 64) - 1) for a in range(len(keys))] for p in range(self.length)]
        self.bins = torch.tensor([[h % 54 for h in row] for row in hashes], device=device)
        self.signs = torch.tensor([[1. if h >> 63 else -1. for h in row] for row in hashes], device=device)
        self.stack_positions = torch.arange(self.length + 4, device=device)
        self.action_ids = torch.arange(len(keys), device=device)
        self.child_positions = torch.arange(4, device=device)

    def initial(self, count: int) -> Prefixes:
        return Prefixes(torch.full((count, self.length), -1, dtype=torch.int64, device=self.device),
            torch.zeros((count, self.length + 4), dtype=torch.int64, device=self.device),
            torch.zeros((count, self.length + 4), dtype=torch.int64, device=self.device),
            torch.ones(count, dtype=torch.int64, device=self.device),
            torch.zeros(count, dtype=torch.int64, device=self.device),
            torch.zeros(count, dtype=torch.int64, device=self.device),
            torch.zeros((count, 54), dtype=torch.float32, device=self.device))

    def encode(self, state: Prefixes, context: Tensor) -> tuple[Tensor, Tensor]:
        count, c = len(state.depth), self.context_size
        top = (state.depth - 1).clamp(min=0)[:, None]
        slot = state.slots.gather(1, top).flatten()
        nest = state.nesting.gather(1, top)
        remaining = (((state.slots <= 1) | (state.slots == 3)) &
                     (self.stack_positions < top)).sum(1)
        need = state.cost[:, None] + remaining[:, None] + self.costs + self.child_costs
        legal = (((self.outputs == slot[:, None]) | ((slot[:, None] == 3) & self.nonzero)) &
                 (~self.temporal | (nest < self.nest_limit)) & (need <= self.cost_limit) &
                 (state.length[:, None] + state.depth[:, None] + self.arities <= self.length))
        active = state.depth > 0
        legal &= active[:, None]
        legal[:, 0] |= ~active
        x = torch.zeros((count, c + 64), dtype=torch.float32, device=self.device)
        x[:, 0], x[:, 1:c + 1] = 1, context
        x[:, c + 1], x[:, c + 2], x[:, c + 3] = state.length / self.length, state.cost / self.cost_limit, state.depth / self.length
        x.scatter_(1, (c + 4 + slot)[:, None], 1)
        x[:, c + 10:] = state.hashed / state.length.clamp(min=1).sqrt()[:, None]
        return x, legal

    def advance(self, state: Prefixes, chosen: Tensor, enabled: Tensor) -> None:
        active = (state.depth > 0) & enabled
        position = state.length.clamp(max=self.length - 1)
        index = position[:, None]
        state.tokens.scatter_(1, index, torch.where(active, chosen, state.tokens.gather(1, index).flatten())[:, None])
        top = (state.depth - 1).clamp(min=0)[:, None]
        nest = state.nesting.gather(1, top) + self.temporal[chosen, None]
        indices = top + self.child_positions
        children = active[:, None] & (self.child_positions < self.arities[chosen, None])
        state.slots.scatter_(1, indices, torch.where(children, self.children[chosen], state.slots.gather(1, indices)))
        state.nesting.scatter_(1, indices, torch.where(children, nest, state.nesting.gather(1, indices)))
        state.hashed.scatter_add_(1, self.bins[position, chosen, None],
                                  (self.signs[position, chosen] * active)[:, None])
        state.depth.add_(torch.where(active, self.arities[chosen] - 1, 0))
        state.cost.add_(self.costs[chosen] * active)
        state.length.add_(active)

    def choose(self, probabilities: Tensor) -> Tensor:
        cumulative = probabilities.cumsum(-1)
        cumulative = cumulative / cumulative[:, -1:]
        draws = torch.rand((len(probabilities), 1), device=self.device, generator=self.rng)
        return torch.searchsorted(cumulative, draws, right=True).flatten()

    def rollout(self, state: Prefixes, context: Tensor, predict: Callable[[Tensor, Tensor], Tensor],
                *, paired: bool = False, record: bool = False) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        features, masks, owners = [], [], []
        rows = torch.arange(len(state.depth), device=self.device)
        enabled = torch.ones_like(state.depth, dtype=torch.bool)
        for step in range(self.length):
            x, legal = self.encode(state, context)
            logp = predict(x, legal)
            if paired:
                probabilities = logp.reshape(-1, 3, logp.shape[-1]).mean(1).exp()
                chosen = self.choose(probabilities).repeat_interleave(3)
            else:
                chosen = self.choose(logp.exp())
            if record:
                features.append(x)
                masks.append(legal)
                owners.append(torch.where(state.depth > 0, rows + step * len(rows), len(rows) * self.length))
            self.advance(state, chosen, enabled)
        return features, masks, owners

    def sample(self, count: int, probes: int, context: Tensor,
               predict: Callable[[Tensor, Tensor], Tensor]) -> Drawn:
        ordinary = self.initial(count)
        features, masks, owners = self.rollout(ordinary, context, predict, record=True)
        pf, pm, pa, pl, pt = [], [], [], [], []
        accepted = 0
        # Each workspace contains at most four prefixes and forty suffix attempts per prefix.
        for attempt in range(0, probes * 24, 4):
            if accepted == probes:
                break
            size = min(4, probes * 24 - attempt)
            source = torch.randint(count, (size,), device=self.device, generator=self.rng)
            positions = (torch.rand(size, device=self.device, generator=self.rng) * ordinary.length[source]).long()
            prefix = ordinary.tokens[source]
            base = self.initial(size)
            for step in range(self.length):
                self.advance(base, prefix[:, step].clamp(min=0), step < positions)
            x, legal = self.encode(base, context)
            p = predict(x, legal).exp()
            first = prefix.gather(1, positions[:, None]).flatten()
            siblings = (self.groups == self.groups[first, None]) & (self.action_ids != first[:, None]) & (p > 1e-4)
            priorities = torch.rand(p.shape, device=self.device, generator=self.rng).masked_fill(~siblings, -1)
            alternatives = priorities.topk(2, dim=1).indices
            actions = torch.cat((first[:, None], alternatives), dim=1)
            eligible = siblings.sum(1) >= 2
            base.depth.mul_(eligible)
            branches = base.select(torch.arange(size, device=self.device).repeat_interleave(120))
            self.advance(branches, actions[:, None, :].expand(-1, 40, -1).reshape(-1),
                         torch.ones_like(branches.depth, dtype=torch.bool))
            self.rollout(branches, context, predict, paired=True)
            tokens = branches.tokens.reshape(size, 40, 3, self.length)
            suffix = tokens[:, :, 0]
            same = (suffix[:, :, None, :] == suffix[:, None, :, :]).all(-1)
            earlier = torch.arange(40, device=self.device)[None, :] < torch.arange(40, device=self.device)[:, None]
            unique = ~(same & earlier).any(-1)
            order = torch.arange(40, device=self.device).expand(size, -1).masked_fill(~unique, 40).sort(-1).values[:, :4]
            good = eligible & (unique.sum(-1) >= 4)
            # Synchronize only one small success vector per probe chunk, never per token.
            keep = [i for i, valid in enumerate(good.cpu().tolist()) if valid][:probes - accepted]
            if not keep:
                continue
            indices = torch.tensor(keep, device=self.device)
            selected = tokens[indices[:, None], order[indices]].reshape(-1, self.length)
            pf.append(x[indices]); pm.append(legal[indices]); pa.append(actions[indices])
            pl.append(positions[indices]); pt.append(selected)
            accepted += len(keep)
        all_features, all_masks = torch.cat(features), torch.cat(masks)
        trace = Trajectories(all_features, all_masks, ordinary.tokens.T.reshape(-1).clamp(min=0),
            torch.stack(owners, dim=1), torch.cat((context.new_ones(1), context)),
            torch.cat(pf) if pf else all_features[:0], torch.cat(pm) if pm else all_masks[:0],
            torch.cat(pa) if pa else torch.empty((0, 3), dtype=torch.int64, device=self.device),
            torch.arange(count, count + 12 * accepted, device=self.device).reshape(-1, 4, 3))
        return Drawn(trace, torch.cat((ordinary.tokens, *pt)),
                     torch.cat(pl) if pl else torch.empty(0, dtype=torch.int64, device=self.device))

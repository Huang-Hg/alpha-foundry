"""FP32 PyTorch policy, trajectory balance and paired-teacher optimization."""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor, nn

from foundry.search.proposer.sampling import Sampler
from foundry.search.proposer.device_sampling import DeviceSampler, Drawn, Trajectories
from foundry.types import HostArray, TrainingResult


class Policy(nn.Module):
    """The original 32-unit tanh policy and context-linear log normalizer."""

    def __init__(self, sampler: Sampler, device: str) -> None:
        super().__init__()
        actions, c = sampler.actions, sampler.context_size
        w1 = sampler.rng.normal(0, .04, (32, c + 64)).astype(np.float32)
        w2 = sampler.rng.normal(0, .04, (len(actions), 33)).astype(np.float32)
        leaves = sum(a.output == 0 and not a.children for a in actions)
        operators = sum(a.output == 0 and bool(a.children) for a in actions)
        for i, action in enumerate(actions):
            if action.output == 0:
                w2[i, -1] = np.log(.65 / max(1, operators) if action.children else .35 / max(1, leaves))
        z = np.zeros(c + 1, dtype=np.float32)
        z[0] = np.log(max(1, leaves)) - 7
        self.w1 = nn.Parameter(torch.tensor(w1, device=device))
        self.w2 = nn.Parameter(torch.tensor(w2, device=device))
        self.z = nn.Parameter(torch.tensor(z, device=device))

    def forward(self, features: Tensor, legal: Tensor) -> Tensor:
        hidden = torch.tanh(features @ self.w1.T)
        logits = hidden @ self.w2[:, :-1].T + self.w2[:, -1]
        return torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)


def trajectory_balance(model: Policy, batch: Trajectories, rewards: Tensor) -> Tensor:
    logp = model(batch.features, batch.legal).gather(1, batch.actions[:, None]).flatten()
    # A padded gather avoids nondeterministic atomic scatter accumulation on CUDA.
    padded = torch.cat((logp, logp.new_zeros(1)))
    residual = padded[batch.owners].sum(1) + model.z @ batch.context - rewards[:len(batch.owners)]
    return residual.square().mean()


def teacher_target(probabilities: Tensor, rewards: Tensor) -> tuple[Tensor, Tensor]:
    """Detached targets (P,3) and reliable-teacher mask (P,), preserving the KL/LCB rule."""
    credit = rewards.mean(1)
    maxima, winners = rewards.max(-1)
    unique = ((rewards - maxima[..., None]).abs() < 1e-10).sum(-1) == 1
    votes = ((winners[..., None] == torch.arange(3, device=rewards.device)) & unique[..., None]).sum(1)
    reliable = votes.max(-1).values >= 3
    centered = credit - credit.max(-1, keepdim=True).values

    def target(alpha: Tensor) -> Tensor:
        value = probabilities * torch.exp(alpha[:, None] * centered)
        return value / value.sum(-1, keepdim=True)

    lo, hi = torch.zeros_like(credit[:, 0]), torch.ones_like(credit[:, 0])
    for _ in range(30):
        mid = (lo + hi) / 2
        q = target(mid)
        kl = (q * (q.log() - probabilities.log())).sum(-1)
        above = kl > .03
        lo, hi = torch.where(above, lo, mid), torch.where(above, mid, hi)
    q = target(lo)
    gain = ((q - probabilities)[:, None, :] * rewards).sum(-1)
    mean = gain.mean(-1)
    lcb = mean - (gain.var(-1, correction=1) / 4).sqrt()
    reliable = reliable & (lcb > 0)
    q = target(lo * (lcb / (mean.abs() + 1e-12)).clamp(0, 1))
    gain = ((q - probabilities)[:, None, :] * rewards).sum(-1)
    reliable = reliable & (gain.mean(-1) - (gain.var(-1, correction=1) / 4).sqrt() > 0)
    return q, reliable


def paired_teacher(model: Policy, batch: Trajectories, rewards: Tensor) -> tuple[Tensor, Tensor]:
    selected = model(batch.probe_features, batch.probe_legal).gather(1, batch.probe_actions)
    conditional = torch.log_softmax(selected, dim=-1)
    with torch.no_grad():
        target, reliable = teacher_target(conditional.exp(), rewards[batch.probe_ids])
    loss = -(target * conditional).sum(-1) * reliable
    return loss.mean(), reliable.sum()


class Learner:
    """Own one model and a dedicated CUDA stream; never borrow account or market buffers."""

    def __init__(self, sampler: Sampler, device: str, seed: int) -> None:
        if device not in ('cpu', 'cuda'):
            raise ValueError('policy device must be cpu or cuda')
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device) if device == 'cuda' else None
        with self.scope():
            self.model = Policy(sampler, device)
            self.sampling = DeviceSampler(sampler, self.device, seed)
            # The public train call sets the requested rate before any update.
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0., foreach=False)
        self.batch: Trajectories | None = None
        self.drawn: Drawn | None = None

    def scope(self) -> torch.cuda.StreamContext | nullcontext:
        return torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()

    def sample(self, count: int, probes: int, context: HostArray) -> tuple[tuple[int, ...], ...]:
        with self.scope(), torch.no_grad():
            self.drawn = self.sampling.sample(count, probes,
                torch.as_tensor(context, dtype=torch.float32, device=self.device), self.model)
            self.batch = self.drawn.trace
            return tuple(tuple(token for token in row if token >= 0) for row in self.drawn.tokens.cpu().tolist())

    def train(self, batch: Trajectories, rewards: HostArray, steps: int, learning_rate: float,
              opd_strength: float) -> TrainingResult:
        with self.scope():
            target = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            model = self.model
            parameters = (model.w1, model.w2)
            optimizer = self.optimizer
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            with torch.no_grad():
                before = trajectory_balance(model, batch, target)
            teachers = target.new_zeros((), dtype=torch.int64)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = trajectory_balance(model, batch, target)
                gradients = torch.autograd.grad(loss, (*parameters, model.z))
                if len(batch.probe_ids):
                    auxiliary, teachers = paired_teacher(model, batch, target)
                    aux = torch.autograd.grad(auxiliary, parameters)
                    tb_norm = sum(g.square().sum() for g in gradients[:2])
                    aux_norm = sum(g.square().sum() for g in aux)
                    scale = (opd_strength * (tb_norm / (aux_norm + 1e-30)).sqrt()).clamp(max=1e4)
                    combined = tuple(g + scale * a for g, a in zip(gradients[:2], aux))
                else:
                    combined = gradients[:2]
                norm = sum(g.square().sum() for g in combined).sqrt()
                clip = (10 / (norm + 1e-12)).clamp(max=1)
                for parameter, gradient in zip(parameters, combined):
                    parameter.grad = gradient * clip
                model.z.grad = gradients[2].clamp(-20, 20)
                optimizer.step()
            with torch.no_grad():
                after = trajectory_balance(model, batch, target)
                values = torch.stack((before, after, teachers, model.z @ batch.context)).cpu().tolist()
            return TrainingResult(values[0], values[1], int(values[2]), values[3])

    def arrays(self) -> dict[str, HostArray]:
        with self.scope():
            return {name: value.detach().cpu().numpy().copy() for name, value in self.model.named_parameters()}

    def restore(self, arrays: dict[str, HostArray], rng: HostArray) -> None:
        with self.scope(), torch.no_grad():
            self.sampling.rng.set_state(torch.from_numpy(rng))
            for name, value in self.model.named_parameters():
                value.copy_(torch.as_tensor(arrays[name], device=self.device))
            self.optimizer.zero_grad(set_to_none=True)
        self.batch = None
        self.drawn = None

    def random_state(self) -> HostArray:
        with self.scope():
            return self.sampling.rng.get_state().numpy().copy()

"""The portfolio objective of the search: rewards from admission against a portfolio frozen for the epoch, a behaviour
archive of recent distinct books, and the joint reweighting of the members (CEM), all on CUDA.

Reward of a candidate: inside admission (a feasible weight exists and its highest positive correlation with the
portfolio or any member is at most rho_max) 1 + 3*sigmoid(utility - .25*co_loss - .1*length - .25*novelty); outside
-6 - min(4, V), V = the admission violation plus the correlation excess plus 1 for an all-cash book; a formula that
fails G1 gets -12. A book equal to an archived one of the same portfolio is a duplicate (-10).
"""
from __future__ import annotations

from collections import OrderedDict

import cupy as cp

from foundry.core import metrics as M
from foundry.portfolio import admission
from foundry.portfolio.library import SEED
from foundry.portfolio.reference import CFG, capped
from foundry.search.policy import CEM

_HASH = cp.RawKernel(r'''
extern "C" __global__ void fingerprints(const double* w, unsigned long long* out, int n) {
    __shared__ unsigned long long hashes[256];
    unsigned long long h=0;
    for(int j=threadIdx.x;j<n;j+=blockDim.x) {
        double v=w[(long long)blockIdx.x*n+j];
        unsigned long long x=v==0?0:__double_as_longlong(v);
        x^=(unsigned long long)(j+1)*0x9e3779b97f4a7c15ULL;
        x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;
        h^=x^(x>>31);
    }
    hashes[threadIdx.x]=h;__syncthreads();
    for(int d=128;d;d>>=1){if(threadIdx.x<d)hashes[threadIdx.x]^=hashes[threadIdx.x+d];__syncthreads();}
    if(threadIdx.x==0)out[blockIdx.x]=hashes[0];
}''', 'fingerprints')


def standardized(r):
    centered = r - r.mean(axis=1, keepdims=True)
    norm = cp.sqrt(cp.sum(centered * centered, axis=1, keepdims=True))
    valid = cp.isfinite(r).all(1) & (norm[:, 0] > 1e-8)
    return cp.where(valid[:, None], centered / cp.maximum(norm, 1e-8), 0), valid


def weight_metrics(rr, dd, ref, ppy):
    """(feasible, utility, deltas) of portfolios with net returns rr (B,T) and max drawdowns dd against the reference:
    the constraints and balanced utility of admission, on the statistics of foundry.core.metrics."""
    delta = cp.stack((M.weighted_sharpe(rr, ref.p, ppy) - ref.wsharpe, M.sharpe(rr, ppy) - ref.sharpe,
                      M.cagr(rr, ppy) - ref.cagr, M.recent_growth(rr, ref.p, ppy) - ref.recent_growth), axis=1)
    feasible = M.sound(rr) & (delta >= admission.return_requirements(dd, ref) - 1e-10).all(1)
    utility = .5 * cp.tanh(delta[:, 0] / CFG.sharpe_scale) + .5 * cp.tanh(delta[:, 3] / CFG.growth_scale)
    return feasible, utility, delta


def member_states(portfolio):
    """(weights, books, net returns (n,T)) of the members, device-resident; the empty library's equal-weight seed
    counts as its one member."""
    engine = portfolio.engine
    members = portfolio.library.members or [SEED]
    formulas = [m['expr'] for m in members if m['expr'] != SEED['expr']]
    known = dict(zip(formulas, engine.members(formulas, portfolio.null_shift)))
    seed = engine.equal_weight()
    states = [(seed, engine.returns([seed])[1][0]) if m['expr'] == SEED['expr'] else known[m['expr']] for m in members]
    return [m['weight'] for m in members], [b for b, _ in states], cp.stack([r for _, r in states])


class Archive:
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.items = OrderedDict()
        self.version = 0

    def update(self, staged):
        for key, value in staged.items():
            self.items[key] = value
            self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        if staged:
            self.version += 1


class Evaluator:
    gate = None          # the portfolio objective admits on the four portfolio margins, not on a single-alpha gate
    COUNTERS = ('requested', 'unique_evaluations', 'g1', 'g2', 'low_corr', 'behavior_duplicates', 'fp64_refined', 'weight_trials')

    def __init__(self, portfolio, archive, rho_max=.9):
        self.portfolio, self.engine, self.archive, self.rho_max = portfolio, portfolio.engine, archive, rho_max
        self.ref = portfolio.reference
        self.version = portfolio.version
        self.archive_version = archive.version
        self.pending = OrderedDict()
        self.cache = {}
        self.best = None
        self.counters = dict.fromkeys(self.COUNTERS, 0)
        self.weights, self.books, self.member_returns = member_states(portfolio)
        self.reference_vectors, self.reference_valid = standardized(cp.concatenate((self.ref.rr[None], self.member_returns)))
        if archive.items:
            self.archive_vectors, _ = standardized(cp.stack([v['rr'] for v in archive.items.values()]))
        else:
            self.archive_vectors = None

    def check_context(self):
        if self.portfolio.version != self.version or self.archive.version != self.archive_version:
            raise RuntimeError('Portfolio/archive changed inside frozen search epoch')

    def context(self):
        r, weights = self.ref.rr, cp.asarray(self.weights)
        # Permutation-invariant member summaries plus chronological portfolio blocks.
        block = cp.minimum(cp.arange(len(r)) * 6 // len(r), 5)
        means = cp.bincount(block, weights=r, minlength=6) / cp.maximum(cp.bincount(block, minlength=6), 1)
        sig = cp.sqrt(cp.mean(r * r))
        member_vol = self.member_returns.std(1)
        values = cp.concatenate((cp.asarray([self.ref.wsharpe / 3, self.ref.sharpe / 3, self.ref.recent_growth, self.ref.cagr]),
                                 cp.stack((cp.sum(weights * weights), cp.sum(weights * member_vol) * self.engine.ppy ** .5,
                                           cp.mean(cp.sum(self.ref.W, axis=1)), cp.asarray(len(self.weights) / 40))),
                                 means / cp.maximum(sig, 1e-8),
                                 cp.stack((cp.asarray(len(self.archive.items) / max(1, self.archive.capacity)),
                                           cp.asarray(0.) if self.archive_vectors is None else
                                           cp.mean(self.archive_vectors @ self.reference_vectors[0])))))
        return cp.asnumpy(cp.tanh(cp.nan_to_num(values))).tolist()

    def evaluate(self, expressions):
        self.check_context()
        self.counters['requested'] += len(expressions)
        fresh = [x for x in dict.fromkeys(expressions) if x not in self.cache]
        for lo in range(0, len(fresh), self.engine.candidate_batch):
            rows = self.engine.formulas(fresh[lo:lo + self.engine.candidate_batch], report=False,
                                        null_shift=self.portfolio.null_shift)
            verdicts = self.portfolio.admit(rows)
            valid = [(r, g) for r, g in zip(rows, verdicts) if g['layers']['G1_structure']['passed']]
            self.counters['unique_evaluations'] += len(rows)
            for r in rows:
                self.cache[r['expr']] = dict(expr=r['expr'], log_reward=-12., passed=False, g1=False,
                                             g2=False, rho=None, utility=None, error=r['error'])
            if valid:
                self._valid_batch(valid)
        return [self.cache[x] for x in expressions]

    def _behavior(self, R):
        """(standardized returns, has-variance flag, highest positive correlation with the portfolio or any
        member, low-correlation flag, similarity to the archive) of a batch of daily returns."""
        vectors, nonzero = standardized(R)
        correlations = cp.clip(vectors @ self.reference_vectors.T, -1, 1)
        rho = cp.maximum(0, cp.where(self.reference_valid[None], correlations, -cp.inf).max(1))
        # An undefined all-cash correlation is never rewarded as novel behavior.
        low_corr = nonzero & cp.isfinite(rho) & (rho <= self.rho_max)
        novelty = cp.zeros(len(R)) if self.archive_vectors is None else cp.maximum(0, vectors @ self.archive_vectors.T).max(1)
        return vectors, nonzero, rho, low_corr, novelty

    def _fingerprints(self, W):
        hashes = cp.empty(len(W), dtype=cp.uint64)
        _HASH((len(W),), (256,), (W, hashes, W.shape[1] * W.shape[2]))
        return cp.asnumpy(hashes).tolist()

    def _duplicate(self, key, book):
        old = self.archive.items.get(key)
        return old is not None and old['portfolio_version'] == self.version and bool(cp.array_equal(book, old['W']))

    def _stage(self, staged, W, R, rows):
        """staged: key -> row of this batch; only the newest `capacity` survive, so only those are copied."""
        for key, i in list(staged.items())[-self.archive.capacity:]:
            self.pending[key] = dict(W=W[i].copy(), rr=R[i].copy(), expr=rows[i]['expr'], portfolio_version=self.version)
            self.pending.move_to_end(key)
            while len(self.pending) > self.archive.capacity:
                self.pending.popitem(last=False)

    def _valid_batch(self, pairs):
        rows, verdicts = zip(*pairs)
        W = cp.stack([r['_W'] for r in rows])
        R = cp.stack([r['_rr'] for r in rows])
        _, nonzero, rho, low_corr, novelty = self._behavior(R)
        down = self.ref.rr < 0
        co_loss = cp.sum(cp.maximum(-R, 0) * down, axis=1) / cp.maximum(cp.sum(cp.abs(R), axis=1), 1e-12)
        proposals = [g['layers']['G2_performance'] for g in verdicts]
        inputs = cp.asarray([[p['passed'], p.get('utility', 0), p.get('violation', 4), len(r['expr']) / 512]
                             for p, r in zip(proposals, rows)])
        feasible = inputs[:, 0].astype(cp.bool_) & low_corr
        distance = inputs[:, 2] + cp.maximum(0, rho - self.rho_max) + (~nonzero)
        raw_reward = cp.where(feasible, 1 + 3 / (1 + cp.exp(-(inputs[:, 1] - .25 * co_loss - .1 * inputs[:, 3] - .25 * novelty))),
                              -6 - cp.minimum(4, distance))
        hash_values = self._fingerprints(W)
        reports = cp.asnumpy(cp.stack((raw_reward, feasible, rho, co_loss, novelty, nonzero), axis=1)).tolist()
        staged = OrderedDict()
        for i, (r, g, p, values, key) in enumerate(zip(rows, verdicts, proposals, reports, hash_values)):
            duplicate = self._duplicate(key, W[i])
            accepted = bool(values[1]) and not duplicate
            report = dict(expr=r['expr'], log_reward=values[0] if not duplicate else -10., passed=accepted,
                          g1=True, g2=p['passed'], rho=values[2], co_loss=values[3], novelty=values[4],
                          duplicate=duplicate, utility=p.get('utility'), weight=p['weight'],
                          delta={k: v[1] for k, v in p['checks'].items()}, precision=p.get('precision'),
                          backend=p['backend'])
            self.cache[r['expr']] = report
            self.counters['g1'] += 1
            self.counters['g2'] += int(p['passed'])
            self.counters['low_corr'] += int(accepted)
            self.counters['behavior_duplicates'] += int(duplicate)
            self.counters['weight_trials'] += p['trials']
            self.counters['fp64_refined'] += int(p['backend'] == 'cuda-f64')
            if accepted and (self.best is None or p['utility'] > self.best[1]['layers']['G2_performance']['utility']):
                # Clones, so neither the row nor its verdict retains the batch allocation behind it.
                row = dict(r, _W=r['_W'].copy(), _rr=r['_rr'].copy())
                self.best = (row, dict(g, layers=dict(g['layers'], G2_performance=dict(p, row=row))))
            if not duplicate and values[5]:
                staged[key] = i
                staged.move_to_end(key)
        self._stage(staged, W, R, rows)

    def accept(self):
        self.check_context()
        if self.best is None:
            return None
        row, previous = self.best
        # Always a fresh FP64 admission, never a rounded reward reused as a verdict.
        proposal = admission.proposals(self.engine, [row], self.ref, precision='f64')[0]
        if not proposal['passed']:
            raise RuntimeError('Selected candidate failed FP64 admission')
        verdict = dict(previous, layers=dict(previous['layers'], G2_performance=proposal))
        weight = self.portfolio.accept(row, verdict)
        return dict(expr=row['expr'], weight=weight, utility=proposal['utility'])

    def reweight(self, iterations=3, count=32, seed=1):
        """Budgeted logit-normal CEM over independent additive weights in (0,1) plus explicit
        deletion proposals, CUDA FP64. A member's own book is never levered; the exposure cap
        is the only coupling between members."""
        self.check_context()
        library = self.portfolio.library
        members = library.members
        if len(members) < 2:
            return None
        books = cp.stack(self.books)
        base = cp.asarray(self.weights, dtype=cp.float64)
        inside = cp.clip(base, 1e-6, 1 - 1e-6)
        sampler = CEM(cp.asnumpy(cp.log(inside / (1 - inside))))
        best = None
        evaluations = 0
        for iteration in range(iterations):
            weights = cp.asarray(sampler.sample(count, seed + iteration))
            deletes = cp.broadcast_to(base, (len(base), len(base))).copy()
            deletes[cp.arange(len(base)), cp.arange(len(base))] = 0
            trials = cp.concatenate((weights, base[None], deletes))
            score_parts = []
            for lo in range(0, len(trials), self.engine.candidate_batch):
                part = trials[lo:lo + self.engine.candidate_batch]
                _, rr, _, dd = self.engine.returns(capped(cp.tensordot(part, books, axes=(1, 0))), 'f64')
                feasible, utility, _ = weight_metrics(rr, dd, self.ref, self.engine.ppy)
                scores = cp.where(feasible, utility, -1e6)
                score_parts.append(scores)
                at = int(cp.argmax(scores))
                if bool(feasible[at]) and (best is None or float(scores[at]) > best[0]):
                    best = (float(scores[at]), part[at].copy())
                evaluations += len(part)
            scores = cp.concatenate(score_parts)
            # No feasible elite: explore again, do not fit arbitrary rejected ties.
            nfeasible = int((scores[:count] > -1e5).sum())
            if nfeasible:
                sampler.update(cp.asnumpy(scores[:count]), min(nfeasible, max(1, count // 4)))
        if best is None:
            return dict(changed=False, evaluations=evaluations)
        self.check_context()
        values = cp.asnumpy(best[1]).tolist()
        library.members = [dict(m, weight=w) for m, w in zip(members, values) if w > 0]
        self.portfolio.changed()
        # One FP64 backtest of the stored portfolio: it is the verdict here and the reference of the next epoch.
        stored = self.portfolio.reference
        feasible, utility, _ = weight_metrics(stored.rr[None], cp.asarray([stored.maxdd]), self.ref, self.engine.ppy)
        if not bool(feasible[0]):
            raise RuntimeError('CEM final portfolio failed FP64 constraints')
        return dict(changed=True, evaluations=evaluations, utility=float(utility[0]), weights=values)

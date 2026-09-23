"""Frozen portfolio objectives, behavior archive and joint weights on CUDA."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tmp/q25_funnel'))
import engine  # Select factor FP64 before loading any CUDA operators.
import cupy as cp
import gates
import combolib
from combo_gpu import capped, launch, return_requirements, static_device
from .native import CEM


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


def version(lib):
    return hashlib.sha256(json.dumps(lib.members or [combolib.SEED], sort_keys=True).encode()).hexdigest()


def standardized(r):
    centered = r - r.mean(axis=1, keepdims=True)
    norm = cp.sqrt(cp.sum(centered * centered, axis=1, keepdims=True))
    valid = cp.isfinite(r).all(1) & (norm[:, 0] > 1e-8)
    return cp.where(valid[:, None], centered / cp.maximum(norm, 1e-8), 0), valid


def weight_metrics(rr, dd, ref):
    """Same net-return constraints and balanced utility as production admission; dd = max drawdowns of the rows."""
    valid = cp.isfinite(rr).all(1) & (rr > -1).all(1)
    p = ref['p']; mu = rr @ p
    sd = cp.sqrt(cp.sum((rr - mu[:, None]) ** 2 * p, axis=1))
    ws = cp.where(sd > 0, mu / cp.maximum(sd, 1e-30) * 365**.5, 0)
    vol = rr.std(1)
    sh = cp.where(vol > 0, rr.mean(1) / cp.maximum(vol, 1e-30) * 365**.5, 0)
    lg = cp.log1p(rr)
    cg = cp.expm1(lg.mean(1) * 365); growth = lg @ p * 365
    delta = cp.stack((ws-ref['wsharpe'], sh-ref['sharpe'], cg-ref['cagr'], growth-ref['recent_growth']), axis=1)
    margin = delta - return_requirements(dd, ref)
    feasible = valid & (margin >= -1e-10).all(1)
    utility = .5*cp.tanh(delta[:, 0]/gates.CFG.sharpe_scale) + .5*cp.tanh(delta[:, 3]/gates.CFG.growth_scale)
    return feasible, utility, delta


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

    def __init__(self, lib, archive, rho_max=.9):
        if not 0 < rho_max <= 1:
            raise ValueError('rho_max must be in (0,1]')
        self.lib, self.archive, self.rho_max = lib, archive, rho_max
        self.ref = lib.ref()
        self.version = version(lib)
        self.archive_version = archive.version
        self.pending = OrderedDict()
        self.cache = {}
        self.best = None
        self.counters = dict.fromkeys(self.COUNTERS, 0)
        states = combolib.member_states(lib.members or [combolib.SEED], 'is')      # device-resident, never recomputed
        self.books = [book for book, _ in states]
        self.member_returns = cp.stack([rr for _, rr in states])
        self.reference_vectors, self.reference_valid = standardized(cp.concatenate((self.ref['rr'][None], self.member_returns)))
        if archive.items:
            self.archive_vectors, _ = standardized(cp.stack([v['rr'] for v in archive.items.values()]))
        else:
            self.archive_vectors = None

    def check_context(self):
        if version(self.lib) != self.version or self.archive.version != self.archive_version:
            raise RuntimeError('Portfolio/archive changed inside frozen search epoch')

    def context(self):
        r = self.ref['rr']; weights = cp.asarray([m['weight'] for m in self.lib.members or [combolib.SEED]])
        # Permutation-invariant member summaries plus chronological portfolio blocks.
        block = cp.minimum(cp.arange(len(r))*6//len(r), 5)
        means = cp.bincount(block, weights=r, minlength=6)/cp.maximum(cp.bincount(block, minlength=6), 1)
        sig = cp.sqrt(cp.mean(r*r))
        member_vol = self.member_returns.std(1)
        values = cp.concatenate((cp.asarray([self.ref['wsharpe']/3, self.ref['sharpe']/3,
                                             self.ref['recent_growth'], self.ref['cagr']]),
                                 cp.stack((cp.sum(weights*weights), cp.sum(weights*member_vol)*365**.5,
                                           cp.mean(cp.sum(self.ref['W'], axis=1)), cp.asarray(len(weights)/40))),
                                 means/cp.maximum(sig, 1e-8),
                                 cp.stack((cp.asarray(len(self.archive.items)/max(1,self.archive.capacity)),
                                           cp.asarray(0.) if self.archive_vectors is None else
                                           cp.mean(self.archive_vectors @ self.reference_vectors[0])))))
        return cp.asnumpy(cp.tanh(cp.nan_to_num(values))).tolist()

    def evaluate(self, expressions):
        self.check_context()
        self.counters['requested'] += len(expressions)
        fresh = [x for x in dict.fromkeys(expressions) if x not in self.cache]
        for lo in range(0, len(fresh), engine.CUDA.candidate_batch):
            batch = fresh[lo:lo+engine.CUDA.candidate_batch]
            rows = engine.simulate_many(batch, report=False)
            verdicts = gates.run_batch(rows, self.ref)
            valid = [(r, g) for r, g in zip(rows, verdicts) if g['layers']['G1_structure']['passed']]
            self.counters['unique_evaluations'] += len(rows)
            for r, g in zip(rows, verdicts):
                self.cache[r['expr']] = dict(expr=r['expr'], log_reward=-12., passed=False, g1=False,
                                            g2=False, rho=None, utility=None, error=r.get('error'))
            if not valid:
                continue
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
        _HASH((len(W),), (256,), (W, hashes, W.shape[1]*W.shape[2]))
        return cp.asnumpy(hashes).tolist()

    def _duplicate(self, key, book):
        old = self.archive.items.get(key)
        return old is not None and old.get('portfolio_version') == self.version and bool(cp.array_equal(book, old['W']))

    def _stage(self, staged, W, R, rows):
        """staged: key -> row of this batch; only the newest `capacity` survive, so only those are copied."""
        for key, i in list(staged.items())[-self.archive.capacity:]:
            self.pending[key] = dict(W=W[i].copy(), rr=R[i].copy(), expr=rows[i]['expr'], portfolio_version=self.version)
            self.pending.move_to_end(key)
            while len(self.pending) > self.archive.capacity:
                self.pending.popitem(last=False)

    def _valid_batch(self, pairs):
        rows, verdicts = zip(*pairs)
        W = cp.stack([r['_W'] for r in rows]); R = cp.stack([r['_rr'] for r in rows])
        _, nonzero, rho, low_corr, novelty = self._behavior(R)
        down = self.ref['rr'] < 0
        co_loss = cp.sum(cp.maximum(-R, 0)*down, axis=1)/cp.maximum(cp.sum(cp.abs(R), axis=1), 1e-12)
        proposals = [g['layers']['G2_performance'] for g in verdicts]
        inputs = cp.asarray([[p['passed'], p.get('utility', 0), p.get('violation', 4), len(r['expr'])/512]
                             for p, r in zip(proposals, rows)])
        feasible = inputs[:, 0].astype(cp.bool_) & low_corr
        distance = inputs[:, 2] + cp.maximum(0, rho-self.rho_max) + (~nonzero)
        raw_reward = cp.where(feasible, 1+3/(1+cp.exp(-(inputs[:, 1]-.25*co_loss-.1*inputs[:, 3]-.25*novelty))),
                              -6-cp.minimum(4, distance))
        hash_values = self._fingerprints(W)
        reports = cp.asnumpy(cp.stack((raw_reward, feasible, rho, co_loss, novelty, nonzero), axis=1)).tolist()
        staged = OrderedDict()
        for i, (r, g, p, values, key) in enumerate(zip(rows, verdicts, proposals, reports, hash_values)):
            duplicate = self._duplicate(key, W[i])
            accepted = bool(values[1]) and not duplicate
            report = dict(expr=r['expr'], log_reward=values[0] if not duplicate else -10., passed=accepted,
                          g1=True, g2=p['passed'], rho=values[2], co_loss=values[3], novelty=values[4],
                          duplicate=duplicate, utility=p.get('utility'), weight=p['weight'],
                          delta={k:v[1] for k,v in p['checks'].items()}, precision=p.get('precision'),
                          backend=p['backend'])
            self.cache[r['expr']] = report
            self.counters['g1'] += 1
            self.counters['g2'] += int(p['passed'])
            self.counters['low_corr'] += int(accepted)
            self.counters['behavior_duplicates'] += int(duplicate)
            self.counters['weight_trials'] += p['trials']
            self.counters['fp64_refined'] += int(p['backend'] == 'cuda-f64')
            if accepted and (self.best is None or p['utility'] > self.best[1]['layers']['G2_performance']['utility']):
                # Clone to avoid retaining every backing batch allocation.
                self.best = (dict(r, _W=r['_W'].copy(), _rr=r['_rr'].copy()), g)
            if not duplicate and values[5]:
                staged[key] = i
                staged.move_to_end(key)
        self._stage(staged, W, R, rows)

    def accept(self):
        self.check_context()
        if self.best is None:
            return None
        row, previous = self.best
        # Always fresh FP64 admission, never reuse a rounded reward as a verdict.
        from combo_gpu import proposal_many
        proposal = proposal_many([row], self.ref, precision='f64')[0]
        if not proposal['passed']:
            raise RuntimeError('Selected candidate failed FP64 admission')
        verdict = dict(previous, layers=dict(previous['layers'], G2_performance=proposal))
        self.lib.add(row, verdict)
        return dict(expr=row['expr'], weight=proposal['weight'], utility=proposal['utility'])

    def reweight(self, iterations=3, count=32, seed=1):
        """Budgeted logit-normal CEM over independent additive weights in (0,1) plus explicit
        deletion proposals, CUDA FP64. A member's own book is never levered; the exposure cap
        is the only coupling between members."""
        self.check_context()
        members = self.lib.members or [combolib.SEED]
        if len(members) < 2:
            return None
        books = cp.stack(self.books)
        base = cp.asarray([m['weight'] for m in members], dtype=cp.float64)
        inside = cp.clip(base, 1e-6, 1-1e-6); logits = cp.log(inside/(1-inside))
        sampler = CEM(cp.asnumpy(logits).tolist())
        best = None
        evaluations = 0
        for iteration in range(iterations):
            buffer = sampler.sample(count, seed+iteration)
            # Host buffer is C++-generated weights, never market arrays or CPU PnL.
            weights = cp.asarray(buffer).view(cp.float64).reshape(count, len(members))
            deletes = cp.broadcast_to(base, (len(base), len(base))).copy()
            deletes[cp.arange(len(base)), cp.arange(len(base))] = 0
            trials = cp.concatenate((weights, base[None], deletes))
            score_parts = []
            for lo in range(0, len(trials), engine.CUDA.candidate_batch):
                part = trials[lo:lo+engine.CUDA.candidate_batch]
                W = capped(cp.tensordot(part, books, axes=(1, 0)))
                W[:, :self.ref['win']['row0']] = 0
                rr, stats = launch(W, static_device(self.ref['win']), self.ref['win']['row0'], 'f64')
                feasible, utility, _ = weight_metrics(rr[:, self.ref['win']['row0']:], stats[:, 3], self.ref)
                scores = cp.where(feasible, utility, -1e6)
                score_parts.append(scores)
                at = int(cp.argmax(scores))
                if bool(feasible[at]) and (best is None or float(scores[at]) > best[0]):
                    best = (float(scores[at]), part[at].copy())
                evaluations += len(part)
            scores = cp.concatenate(score_parts)
            # No feasible elite: explore again, do not fit arbitrary rejected ties.
            if bool((scores[:count] > -1e5).any()):
                nfeasible = int((scores[:count] > -1e5).sum())
                sampler.update(cp.asnumpy(scores[:count]).tolist(), min(nfeasible, max(1, count//4)))
        if best is None:
            return dict(changed=False, evaluations=evaluations)
        self.check_context()
        values = cp.asnumpy(best[1]).tolist()
        self.lib.members = [dict(m, weight=w) for m, w in zip(members, values) if w > 0]
        self.lib._ref.clear()
        # One FP64 backtest of the stored portfolio: it is the verdict here and the reference of the next epoch.
        stored = self.lib.ref()
        feasible, utility, _ = weight_metrics(stored['rr'][None], cp.asarray([stored['maxdd']]), self.ref)
        if not bool(feasible[0]):
            raise RuntimeError('CEM final portfolio failed FP64 constraints')
        return dict(changed=True, evaluations=evaluations, utility=float(utility[0]), weights=values)

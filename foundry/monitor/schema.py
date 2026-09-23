"""The dashboard's input contract: every field its producers write, declared as typed dictionaries.

  SearchStatus   foundry/search/run.py writes <run>/status.json and mirrors it to tmp/monitor/live.json (+ `run`).
  ServiceStatus  foundry/service/daemon.py Server._status() writes tmp/service/status.json (kind='service').

Field names are the producers'; components read a status only through these declarations. A search status grows
over the run (a loading run has written a handful of fields), so every key is optional and an absent one means
"not reported yet".

A numeric field's annotation also says how the watcher (watch.py) moves it between snapshots: GLIDES readings move
smoothly from the value on screen to the reported one; RESTARTS ones only grow while one run or one service lasts,
so one going backwards is a new run and is shown at once. GLIDE and RESTART below are derived from the annotations,
so declaring a field is all it takes to animate it.
"""
from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, TypedDict, Union, get_args, get_origin, get_type_hints

GLIDES, RESTARTS = 'glides', 'restarts'
GlidingInt = Annotated[int, GLIDES]
GlidingFloat = Annotated[float, GLIDES]
RunCounter = Annotated[int, GLIDES, RESTARTS]

Ended = Literal['complete', 'failed', 'stopped', 'interrupted']        # a run writes nothing after these
SearchPhase = Literal['loading', 'sampling', 'evaluating', 'training', 'reweighting', 'checkpoint', Ended]
ENDED = get_args(Ended)


class Source(TypedDict, total=False):
    """Added by the reader (watch.py, preview.py), not by a producer."""
    _source_mtime: float                     # when the status file was written, local clock
    _source_name: str                        # its file name


# ------------------------------------------------------------------ search run

class Delta(TypedDict, total=False):
    """What a candidate changes. Portfolio objective: the admission checks of foundry/portfolio/admission.py (KEYS,
    plus capacity once the portfolio is full). Single objective: its own Sharpe above its gate, and its own CAGR."""
    sharpe: Optional[float]                  # None: the single-alpha gate is closed
    full_sharpe: float
    cagr: float
    recent_growth: float
    capacity: float


class Precision(TypedDict, total=False):
    """How an admission was computed under CUDA mixed precision (admission.py)."""
    mode: str
    refined: bool
    calibration_error: float
    return_envelope: float
    over_budget: bool
    unstable: bool
    fp64_trials: int
    fp32_trials: int


class CandidateReport(TypedDict, total=False):
    """One evaluated expression; the status keeps the newest eight. A candidate that fails G1 carries only
    expr, log_reward, passed, g1, g2 (g3), rho, utility and error."""
    expr: str
    log_reward: float
    passed: bool                             # admissible: every gate passed, low correlation, not a duplicate
    g1: bool
    g2: bool
    g3: bool                                 # single objective
    rho: Optional[float]
    co_loss: float                           # portfolio objective
    novelty: float
    duplicate: bool
    utility: Optional[float]
    weight: float
    delta: Delta
    precision: Optional[Precision]           # portfolio objective
    backend: str
    error: Optional[str]
    # Single objective: the candidate's own book, and the Sharpe its correlations require.
    sharpe: float
    cagr: float
    maxdd: float
    rho_portfolio: float
    rho_members: float
    appraisal: float
    required_sharpe: Optional[float]         # None: the gate is closed
    recent_years: List[float]


class Gate(TypedDict, total=False):
    """The single-alpha objective's admission gate, frozen for one epoch."""
    prior_mean: float
    shrinkage: float
    reference_sharpe: float
    required_appraisal: Optional[float]      # None: the gate is closed
    anchor_correlation: float
    anchor_sharpe: float
    pool_size: int
    pool_target: float
    recent_years: int
    recent_year_sharpe: float


class SearchStatus(Source, total=False):
    run: str                                 # run directory; the live mirror adds it
    market: str                              # shown after the title when a producer names its market
    phase: SearchPhase
    error: str                               # failed / interrupted
    mode: Literal['gfn', 'random']
    objective: str                           # key of objectives.OBJECTIVES; runs that predate it are 'portfolio'
    null_shift: bool                         # noise control: every signal delayed
    device: str
    epoch: RunCounter
    epochs: int
    elapsed_seconds: float
    # The funnel, cumulative over the run: each counter counts a subset of the one above it.
    generated: int
    requested: GlidingInt
    unique_evaluations: RunCounter
    g1: GlidingInt
    g2: GlidingInt
    g3: GlidingInt                           # single objective
    low_corr: GlidingInt
    accepted: GlidingInt
    behavior_duplicates: GlidingInt
    fp64_refined: GlidingInt
    weight_trials: GlidingInt
    probe_evaluations: GlidingInt
    reweight_evaluations: GlidingInt
    recent: List[CandidateReport]
    gate: Optional[Gate]                     # None under the portfolio objective
    initial_sharpe: float
    initial_cagr: float
    portfolio_sharpe: GlidingFloat
    portfolio_cagr: GlidingFloat
    # Performance: batch sizes, the last epoch's stage timings and learning, throughput.
    candidate_batch: int
    trial_batch: int
    gpu_used_mib: GlidingInt
    sampling_seconds: float
    evaluation_seconds: float
    training_seconds: float
    reweight_seconds: float
    evaluation_candidates_per_second: GlidingFloat
    candidates_per_second: GlidingFloat
    loss_before: float
    loss_after: float
    teachers: int
    logz: float


# ------------------------------------------------------------------ evaluation service

Op = Literal['eval', 'family']


class PortfolioSummary(TypedDict, total=False):
    """The portfolio reference the service admits against."""
    n: int
    sharpe: float
    cagr: float
    maxdd: float


class CurrentJob(TypedDict, total=False):
    """The job on the GPU (several coalesced evaluation requests count as one)."""
    op: Op
    clients: int
    t0: float                                # service clock
    source: Optional[str]                    # submitted file; its directory names the submitter
    n: int
    # From the first progress tick on:
    stage: str
    stage_t0: float
    done: int
    total: int


class JobRecord(TypedDict, total=False):
    """A finished job; the service keeps the last 32."""
    t: float                                 # service clock
    op: Op
    source: Optional[str]
    dt: float
    ok: bool
    n: int
    variants: int
    verdicts: Dict[str, int]


class SubmitterTotals(TypedDict, total=False):
    """Every round one submitting directory has sent since the service started."""
    rounds: int
    formulas: int
    variants: int
    verdicts: Dict[str, int]
    last: float                              # service clock


class ServiceStatus(Source, total=False):
    kind: Literal['service']
    market: str
    phase: Literal['warm', 'serve']
    warm: str                                # what warm-up is doing
    pid: int
    now: float                               # service clock when written; only differences to it are shown
    elapsed_seconds: float
    busy_seconds: float
    device: Optional[str]                    # None while warming up
    vram: str
    queue: int
    clients: int
    served: int
    exprs: RunCounter
    batches: int
    portfolio: Optional[PortfolioSummary]
    cur: Optional[CurrentJob]
    jobs: List[JobRecord]
    sources: Dict[str, SubmitterTotals]


Status = Union[SearchStatus, ServiceStatus]


def _marked(marker):
    return tuple(dict.fromkeys(key for status in (SearchStatus, ServiceStatus)
                               for key, hint in get_type_hints(status, include_extras=True).items()
                               if get_origin(hint) is Annotated and marker in hint.__metadata__))


GLIDE, RESTART = _marked(GLIDES), _marked(RESTARTS)

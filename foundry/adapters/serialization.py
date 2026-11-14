"""JSON conversion at the application boundary; device buffers never cross it."""
from dataclasses import asdict, dataclass, is_dataclass
import math
import types
from typing import Literal, get_args, get_origin, get_type_hints
from foundry.candidates import ArraySource, EqualWeight, Formula, Source
from foundry.ops.registry import OPS, ARITY, COST, MIN_WINDOW, MAX_WINDOW
from foundry.portfolio.library import PROTOCOL
from foundry.research_types import Candidate, CommitResult, Metrics, ResearchContext, RiskSummary, Verdict


REPORT_NOTES = {
    Metrics: {
        'turnover': 'Mean over consecutive research-window days of sum(abs(target[t]-target[t-1])); target-book fraction per day, not executed notional.',
        'exposure': 'Research-window mean of sum(abs(daily target book)); target notional/equity ratio, not actual account leverage.',
        'max_single_weight': 'Maximum absolute daily target-book value over the research window; target notional/equity ratio.',
        'margin_bps': 'ann_ret / max(turnover * periods_per_year, 1e-12) * 1e4; basis points per annualized target-turnover proxy, not realized margin or execution profit per trade.',
    },
    RiskSummary: {
        'model': 'Stable execution-account model identifier, including collateral timing and liquidation assumptions.',
        'funding': 'Research-window sum in initial-NAV units (initial NAV=1); positive is funding received, negative is funding paid.',
        'trading_fees': 'Research-window sum of actual nonnegative trading charges in initial-NAV units.',
        'liquidation_fees': 'Research-window sum of actual nonnegative liquidation charges in initial-NAV units; limited to available isolated-bucket equity.',
        'isolated_loss_cap': 'Research-window sum of max(-isolated bucket equity,0) at liquidation, in initial-NAV units. This nonnegative accounting adjustment protects free cash under the model; it is not verified exchange insurance cashflow.',
        'liquidations': 'Research-window total count of per-asset liquidation events across all execution stages; not the number of affected hours.',
        'first_liquidation_ms': 'Full-run first liquidation event-bar timestamp in UTC milliseconds; -1 means none. Intrabar events identify the containing hour, not the exact touch or fill time.',
        'bankrupt': 'Full-run account bankruptcy flag; not an isolated-position liquidation flag.',
        'ending_equity': 'Final full-run account equity in initial-NAV units (initial NAV=1).',
        'max_gross_leverage': 'Maximum full-run actual gross notional/account equity ratio sampled at mark open before liquidation, after daily rebalance, and at mark close; not an intrabar maximum. Adverse prices can exceed the entry limit; infinity at nonpositive equity serializes as null.',
    },
}


@dataclass(frozen=True)
class CandidateReport:
    id: str
    source: Source
    ok: bool
    error: str | None
    precision: str
    metrics: Metrics | None
    risk: RiskSummary | None
    decision: Verdict
    accepted: bool
    weight: float


@dataclass(frozen=True)
class EvaluationReport:
    context: ResearchContext
    rows: tuple[CandidateReport, ...]


@dataclass(frozen=True)
class CommitReport:
    before: ResearchContext
    context: ResearchContext
    rows: tuple[CandidateReport, ...]


@dataclass(frozen=True)
class RegisteredBatch:
    context: ResearchContext
    candidates: tuple[Candidate, ...]


def source_from_json(record):
    factory = {'formula': Formula, 'signals': ArraySource, 'target_weights': ArraySource, 'target_exposure': ArraySource,
               'equal_weight': EqualWeight}[record['kind']]
    return factory(**record)


def candidate_from_json(record):
    return Candidate(record['id'], source_from_json(record['source']))


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def candidate_report(result):
    row = result.row
    return CandidateReport(result.id, row.source, row.ok, row.error, row.precision, row.metrics, row.risk,
                           result.verdict, result.accepted, result.weight)


def candidate_result(result):
    return json_value(asdict(candidate_report(result)))


def evaluation_result(result):
    rows = tuple(candidate_report(row) for row in result.rows)
    report = CommitReport(result.before, result.context, rows) if isinstance(result, CommitResult) else EvaluationReport(result.context, rows)
    return json_value(asdict(report))


def schema_for(kind):
    """Describe the same dataclasses that the JSON serializer publishes."""
    origin, args = get_origin(kind), get_args(kind)
    if is_dataclass(kind):
        properties = {name: schema_for(annotation) for name, annotation in get_type_hints(kind).items()}
        schema = dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
        if kind in REPORT_NOTES:
            schema['description'] = kind.__doc__.strip()
            for name, note in REPORT_NOTES[kind].items():
                properties[name]['description'] = note
        return schema
    if origin is types.UnionType:
        return dict(anyOf=[schema_for(item) for item in args])
    if origin is Literal:
        return dict(enum=list(args))
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return dict(type='array', items=schema_for(args[0]))
        return dict(type='array', prefixItems=[schema_for(item) for item in args], minItems=len(args), maxItems=len(args))
    return {str: dict(type='string'), bool: dict(type='boolean'), int: dict(type='integer'),
            float: dict(type=['number', 'null']), type(None): dict(type='null')}[kind]


def describe(research):
    engine = research.engine
    return dict(protocol=PROTOCOL, context=asdict(research.context), precision=asdict(engine.context.precision),
        allocation=asdict(engine.allocation),
        metrics=schema_for(Metrics),
        fields=[dict(name=name, category=engine.panel.fields[name].category, scope=engine.panel.fields[name].scope,
                     description=engine.panel.fields[name].doc) for name in engine.vocabulary],
        operators=[dict(name=op.name, kind=op.kind, arity=ARITY[op.kind], cost=COST[op.kind], description=op.doc)
                   for op in OPS.values()], window=dict(min=MIN_WINDOW, max=MAX_WINDOW), batch_size=engine.candidate_batch,
        axes=dict(snapshot=engine.context.id, dates=[str(day) for day in engine.dates], assets=list(engine.panel.assets)),
        external_batch=dict(kinds=['signals', 'target_exposure'] if engine.allocation.allow_short else ['signals', 'target_weights', 'target_exposure'],
            values='(B,T,S) float32 or float64; exact published axes',
            ids='B unique candidate IDs', snapshot='context.snapshot.id', producer='nonempty producer identity',
            signals=('NaN is missing; clip(-1,1)/eligible_count gives signed exposure at at most 1x gross.'
                     if engine.allocation.allow_short else 'NaN is missing; clip(0,1)/eligible_count.'),
            target_weights='Cash markets only: finite nonnegative fractions, zero for ineligible assets, daily sum <= 1.',
            target_exposure='Notional/equity ratios: signed only when allocation.allow_short, abs-sum <= allocation.max_gross; zero for ineligible assets.',
            weight_sum_atol=engine.portfolio_cfg.weight_sum_atol,
            rounding='Gross exposure above allocation.max_gross within weight_sum_atol is normalized to the budget; larger excess is rejected.',
            observed_through='(B,T) int64; last source row known to the producer, -1 <= value <= decision row',
            fit_through='(B,T) int64; last fit row used by the producer, -1 <= value <= decision row',
            pit='Producer-declared cutoffs are checked for forward references; the engine cannot prove the provenance of externally computed values.',
            persistence='research.register_batch writes immutable artifacts; evaluate and commit only read registered sources'),
        execution_risk=dict(report=schema_for(RiskSummary),
            small_perpetual_cpu_cells=engine.cuda_cfg.perpetual_host_cells,
            routing='Perpetual batches at or below B*T*S cells are planned on CPU before execution; larger batches use the selected device. No precision replay.',
            units='Initial NAV=1; funding is signed cashflow, negative means payment. Nonfinite numeric scalars serialize as null.',
            isolated_loss_cap='Accounting adjustment max(-isolated bucket equity,0) at liquidation; this is the isolated-loss model assumption, not verified insurance cashflow.',
            events=dict(path='SimulationResult.risk.liquidation_events', shape=['batch', 'execution_event', 'asset'],
                dtype='uint8', encoding='bitwise_or', stages=dict(open=1, funding=2, rebalance=4, intrabar=8),
                coverage='Full input grid; this model uses hourly event bars. The scalar JSON risk report aggregates counts and omits the bitmap.'),
            cash='risk=null explicitly denotes cash execution without derivatives risk accounting.'))

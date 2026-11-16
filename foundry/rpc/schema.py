"""Typed JSON-RPC schemas and protocol limits; importing opens no resources."""
from dataclasses import MISSING, asdict, fields, is_dataclass
from collections.abc import Mapping
import types
from typing import Literal, get_args, get_origin, get_type_hints, is_typeddict

from foundry.config import ROOT, PARAMETER_GROUPS, Parameters, configure
from foundry.portfolio.library import PROTOCOL
from foundry.types import (
    Candidate, CandidateGrid, CommitReport, Diagnostics, EvaluationReport, FamilyResult,
    FamilySeed, Metrics, PortfolioReport, RegisteredBatch, ResearchContext, ReweightResult,
    RiskSummary, WeightCommit, WeightReport,
)

HOST = '127.0.0.1'
STATE = ROOT / 'var' / 'service'
PORTFILE = STATE / 'port.json'
LOCKFILE = STATE / 'start.lock'
LOGFILE = STATE / 'service.log'
STATUSFILE = STATE / 'status.json'
MAX_ARRAY_BYTES = 1 << 30
MAX_REQUEST_BYTES = 1 << 20
MAX_RESPONSE_BYTES = 1 << 30
MAX_PENDING = 64
HEARTBEAT_SECONDS = 10.
BUNDLE = 'candidate-batch-v3'
CHECKPOINT_FORMAT = 'candidate-discovery-v5'

REPORT_NOTES = {
    Metrics: {
        'turnover': 'Mean over consecutive research-window decisions of sum(abs(target[t]-target[t-1])); target-book fraction per decision, not executed notional.',
        'exposure': 'Research-window mean of sum(abs(target book sampled at each statistics close)); target notional/equity ratio, not actual account leverage.',
        'max_single_weight': 'Maximum absolute decision target-book value over the research window; target notional/equity ratio.',
        'margin_bps': 'ann_ret / max(turnover * decision_periods_per_year, 1e-12) * 1e4; basis points per annualized target-turnover proxy, not realized margin or execution profit per trade.',
    },
    RiskSummary: {
        'model': 'Stable execution-account model identifier, including collateral timing and liquidation assumptions.',
        'funding': 'Research-window sum in initial-NAV units (initial NAV=1); positive is funding received, negative is funding paid.',
        'trading_fees': 'Research-window sum of actual nonnegative trading charges in initial-NAV units.',
        'liquidation_fees': 'Research-window sum of actual nonnegative liquidation charges in initial-NAV units; limited to available isolated-bucket equity.',
        'isolated_loss_cap': 'Research-window sum of max(-isolated bucket equity,0) at liquidation, in initial-NAV units. This nonnegative accounting adjustment protects free cash under the model; it is not verified exchange insurance cashflow.',
        'liquidations': 'Research-window total count of per-asset liquidation events across all execution stages; not the number of affected base bars.',
        'first_liquidation_ns': 'Full-run first liquidation event-bar timestamp in UTC nanoseconds; -1 means none. Intrabar events identify the containing base bar, not the exact touch or fill time.',
        'bankrupt': 'Full-run account bankruptcy flag; not an isolated-position liquidation flag.',
        'ending_equity': 'Final full-run account equity in initial-NAV units (initial NAV=1).',
        'max_gross_leverage': 'Maximum full-run actual gross notional/account equity ratio sampled at mark open before liquidation, after target update, and at mark close; not an intrabar maximum. Adverse prices can exceed the entry limit; infinity at nonpositive equity serializes as null.',
    },
}


def schema_for(kind):
    """Describe the same dataclasses that the JSON serializer publishes."""
    origin, args = get_origin(kind), get_args(kind)
    if is_typeddict(kind):
        properties = {}
        for name, annotation in get_type_hints(kind).items():
            properties[name] = schema_for(annotation)
            if is_dataclass(annotation):
                properties[name]['required'] = [f.name for f in fields(annotation)
                    if f.default is MISSING and f.default_factory is MISSING]
        return dict(type='object', properties=properties, required=sorted(kind.__required_keys__), additionalProperties=False)
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
    if origin is Mapping:
        return dict(type='object', additionalProperties=schema_for(args[1]))
    if kind is bytes:
        return dict(type='string', contentEncoding='base64')
    if origin is Literal:
        return dict(enum=list(args))
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return dict(type='array', items=schema_for(args[0]))
        return dict(type='array', prefixItems=[schema_for(item) for item in args], minItems=len(args), maxItems=len(args))
    return {str: dict(type='string'), bool: dict(type='boolean'), int: dict(type='integer'),
            float: dict(type=['number', 'null']), type(None): dict(type='null')}[kind]


def parameter_values(settings):
    """Expose the resolved public parameter names, rather than internal assembly fields."""
    return dict(market=settings.market.name, backend=settings.backend,
                **{key: value for key, value in asdict(settings.market).items() if key != 'name'},
                **{name: asdict(getattr(settings, field)) for name, (field, _) in PARAMETER_GROUPS.items()})


def object_schema(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


TEXT = dict(type='string', minLength=1)
CANDIDATE = schema_for(Candidate)
CANDIDATE['properties']['id'] = TEXT
FAMILY = schema_for(FamilySeed | CandidateGrid)
METHODS = {
    'system.status': object_schema({}),
    'system.stop': object_schema({}),
    'research.describe': object_schema({}),
    'research.refresh': object_schema({}),
    'research.register_batch': object_schema(dict(context=TEXT,
        bytes=dict(type='integer', minimum=1, maximum=MAX_ARRAY_BYTES))),
    'research.retain': object_schema(dict(context=TEXT, candidates=dict(type='array', minItems=1, items=CANDIDATE))),
    'research.release_batch': object_schema(dict(lease=TEXT)),
    'research.evaluate': object_schema(dict(context=TEXT, candidates=dict(type='array', minItems=1, items=CANDIDATE))),
    'research.commit': object_schema(dict(context=TEXT, candidates=dict(type='array', minItems=1, items=CANDIDATE))),
    'research.diagnose': object_schema(dict(context=TEXT)),
    'research.judge_family': object_schema(dict(context=TEXT, families=dict(type='array', minItems=1, items=FAMILY))),
    'research.portfolio': object_schema(dict(context=TEXT)),
    'research.trial_weights': object_schema(dict(context=TEXT, weights=dict(type='array', minItems=1,
        items=dict(type='array', minItems=1, items=dict(type='number', minimum=0))))),
    'research.commit_weights': object_schema(dict(context=TEXT,
        weights=dict(type='array', minItems=1, items=dict(type='number', minimum=0)))),
    'research.reweight': object_schema(dict(context=TEXT, iterations=dict(type='integer', minimum=0, maximum=100),
        count=dict(type='integer', minimum=2, maximum=8192), seed=dict(type='integer', minimum=0))),
}
REQUEST = object_schema(dict(jsonrpc=dict(const='2.0'), id=TEXT,
    method=dict(enum=list(METHODS)), params=dict(type='object')))
REQUEST.update({'$schema': 'https://json-schema.org/draft/2020-12/schema',
    'allOf': [{'if': {'properties': {'method': {'const': method}}},
               'then': {'properties': {'params': parameters}}} for method, parameters in METHODS.items()]})
RESPONSES = {
    'research.refresh': schema_for(ResearchContext),
    'research.evaluate': schema_for(EvaluationReport),
    'research.commit': schema_for(CommitReport),
    'research.register_batch': schema_for(RegisteredBatch),
    'research.retain': schema_for(RegisteredBatch),
    'research.release_batch': schema_for(type(None)),
    'research.diagnose': schema_for(Diagnostics),
    'research.judge_family': schema_for(FamilyResult),
    'research.portfolio': schema_for(PortfolioReport),
    'research.trial_weights': schema_for(WeightReport),
    'research.commit_weights': schema_for(WeightCommit),
    'research.reweight': schema_for(ReweightResult),
}


def parse(request):
    """Validate one external request before enqueueing or opening a transport."""
    from jsonschema import ValidationError, validate

    try:
        validate(request, REQUEST)
    except ValidationError as error:
        raise ValueError(error.message) from error
    if request['method'] in ('research.evaluate', 'research.commit', 'research.retain'):
        ids = [row['id'] for row in request['params']['candidates']]
        if len(ids) != len(set(ids)):
            raise ValueError('candidate IDs must be unique within a request')
    if request['method'] == 'research.judge_family':
        ids = [row['id'] if 'id' in row else row['seed']['id'] for row in request['params']['families']]
        if any(not identity for identity in ids) or len(ids) != len(set(ids)):
            raise ValueError('family IDs must be nonempty and unique within a request')
    return request


def result(request, value):
    return dict(jsonrpc='2.0', id=request['id'], result=value)


def validate_response(method: str, value) -> None:
    """Validate typed reports once when they enter a client process."""
    from jsonschema import ValidationError, validate

    if method in RESPONSES:
        try:
            validate(value, RESPONSES[method])
        except ValidationError as error:
            raise ValueError(error.message) from error


def describe():
    defaults, variants = {}, []
    parameters = schema_for(Parameters)
    for market in ('us_equity', 'a_share', 'crypto_spot', 'crypto_perpetual'):
        settings = configure(market=market)
        defaults[market] = parameter_values(settings)
        allowed = set(defaults[market])
        variant = dict(parameters, properties={name: value for name, value in parameters['properties'].items()
                                               if name in allowed})
        variant['properties']['market'] = dict(const=market)
        variant['required'] = [] if market == 'us_equity' else ['market']
        variants.append(variant)
    return dict(protocol=PROTOCOL, request=REQUEST, methods=METHODS,
        parameters=dict(oneOf=variants), defaults=defaults, responses=RESPONSES,
        workflow=['research.describe -> context.id',
                  'research.register_batch(context, bytes) + binary bundle -> resident candidates and lease',
                  'research.evaluate(context, candidates)',
                  'research.diagnose(context) / research.judge_family(context, families)',
                  'research.commit(context, candidates) / research.reweight(context, iterations, count, seed) -> new context.id',
                  'research.retain(context, candidates) -> durable references; research.release_batch(lease) -> null',
                  'research.refresh -> current context after an explicit conflict'],
        binary='register_batch carries exactly params.bytes bytes after the JSON request newline; 1GiB encoded and decoded limits; no pickle',
        python='with foundry.open() as session: report = session.evaluate(formulas); session.commit(report)')

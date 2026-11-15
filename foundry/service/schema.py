"""JSON-RPC contract. Validation occurs only at the external request boundary."""
from jsonschema import validate
from foundry.config import Parameters, configure
from foundry.portfolio.library import PROTOCOL
from foundry.types import FamilyResult
from foundry.types import (Candidate, CommitReport, EvaluationReport, FamilySeed, PortfolioReport,
                                   RegisteredBatch, WeightCommit, WeightReport)
from foundry.adapters.serialization import parameter_values, schema_for

MAX_ARRAY_BYTES = 1 << 30


def object_schema(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


TEXT=dict(type='string',minLength=1)
CANDIDATE=schema_for(Candidate)
SEED=schema_for(FamilySeed)
for contract in (CANDIDATE, SEED):
    for field in contract['properties'].values():
        if field.get('type') == 'string':
            field['minLength'] = 1
METHODS={
    'system.status':object_schema({}),
    'system.stop':object_schema({}),
    'research.describe':object_schema({}),
    'research.register_batch':object_schema(dict(context=TEXT, bytes=dict(type='integer', minimum=1, maximum=MAX_ARRAY_BYTES))),
    'research.evaluate':object_schema(dict(context=TEXT,candidates=dict(type='array',minItems=1,items=CANDIDATE))),
    'research.commit':object_schema(dict(context=TEXT,candidates=dict(type='array',minItems=1,items=CANDIDATE))),
    'research.family':object_schema(dict(context=TEXT,seeds=dict(type='array',minItems=1,items=SEED))),
    'research.portfolio':object_schema(dict(context=TEXT)),
    'research.trial_weights':object_schema(dict(context=TEXT, weights=dict(type='array', minItems=1,
        items=dict(type='array', minItems=1, items=dict(type='number', minimum=0))))),
    'research.commit_weights':object_schema(dict(context=TEXT, weights=dict(type='array', minItems=1,
        items=dict(type='number', minimum=0)))),
}
REQUEST=object_schema(dict(jsonrpc=dict(const='2.0'),id=TEXT,method=dict(enum=list(METHODS)),params=dict(type='object')))
REQUEST.update({'$schema':'https://json-schema.org/draft/2020-12/schema',
                'allOf':[{'if':{'properties':{'method':{'const':method}}},
                          'then':{'properties':{'params':parameters}}} for method,parameters in METHODS.items()]})


def parse(request):
    validate(request,REQUEST)
    if request['method'] in ('research.evaluate','research.commit','research.family'):
        key='candidates' if request['method']!='research.family' else 'seeds'
        ids=[row['id'] for row in request['params'][key]]
        if len(ids)!=len(set(ids)):
            raise ValueError('candidate IDs must be unique within a request')
    return request


def result(request,value):
    return dict(jsonrpc='2.0',id=request['id'],result=value)


def describe():
    defaults = {}
    variants = []
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
    return dict(protocol=PROTOCOL,request=REQUEST,methods=METHODS,
                parameters=dict(oneOf=variants), defaults=defaults,
                responses={'research.evaluate':schema_for(EvaluationReport),
                           'research.commit':schema_for(CommitReport),
                           'research.register_batch':schema_for(RegisteredBatch),
                           'research.family':schema_for(FamilyResult),
                           'research.portfolio':schema_for(PortfolioReport),
                           'research.trial_weights':schema_for(WeightReport),
                           'research.commit_weights':schema_for(WeightCommit)},
                workflow=['research.describe -> context.id',
                          'research.register_batch(context, bytes) + binary bundle -> typed candidates',
                          'research.evaluate(context, candidates)',
                          'research.commit(context, candidates) -> new context.id'],
                binary='register_batch carries exactly params.bytes bytes after the JSON request newline; no server filesystem path is exposed',
                python='with foundry.open() as session: report = session.evaluate(formulas); session.commit(report)')

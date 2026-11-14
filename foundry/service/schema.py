"""JSON-RPC contract. Validation occurs only at the external request boundary."""
from jsonschema import validate
from foundry.portfolio.library import PROTOCOL
from foundry.portfolio.family import FamilyResult
from foundry.research_types import Candidate, FamilySeed
from foundry.adapters.serialization import CommitReport, EvaluationReport, RegisteredBatch, schema_for


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
    'research.register_batch':object_schema(dict(context=TEXT,path=TEXT)),
    'research.evaluate':object_schema(dict(context=TEXT,candidates=dict(type='array',minItems=1,items=CANDIDATE))),
    'research.commit':object_schema(dict(context=TEXT,candidates=dict(type='array',minItems=1,items=CANDIDATE))),
    'research.family':object_schema(dict(context=TEXT,seeds=dict(type='array',minItems=1,items=SEED))),
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
    return dict(protocol=PROTOCOL,request=REQUEST,methods=METHODS,
                responses={'research.evaluate':schema_for(EvaluationReport),
                           'research.commit':schema_for(CommitReport),
                           'research.register_batch':schema_for(RegisteredBatch),
                           'research.family':schema_for(FamilyResult)},
                workflow=['research.describe -> context.id',
                          'research.register_batch(context, path) -> typed candidates for external arrays',
                          'research.evaluate(context, candidates)',
                          'research.commit(context, candidates) -> new context.id'],
                python='Research(engine, store, arrays).evaluate_device(sources, context_id, report=False) -> DeviceBatch')

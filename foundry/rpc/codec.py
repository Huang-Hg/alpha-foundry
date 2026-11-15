"""Strict JSON and bounded NumPy wire codecs; device execution stays below RPC."""
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from io import BytesIO
import base64
import json
import math
from pathlib import Path
import types
from typing import get_args, get_origin, get_type_hints
from zipfile import BadZipFile, ZipFile

import numpy as np

from foundry.rpc.schema import (
    BUNDLE, CHECKPOINT_FORMAT, MAX_ARRAY_BYTES, parameter_values, schema_for,
)
from foundry.ops.registry import OPS, ARITY, COST, MIN_WINDOW, MAX_WINDOW
from foundry.portfolio.library import PROTOCOL
from foundry.study.session import candidate_report, evaluation_report
from foundry.study.store import encode_archive, decode_archive, publish, source_from_json
from foundry.types import (
    ArrayBatch, Candidate, Metrics, MiningCheckpoint, MiningEpoch, MiningStatus,
    RiskSummary, Source,
)


def candidate_from_json(record):
    return from_json(Candidate, record)


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode('ascii')
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def candidate_result(result):
    return json_value(asdict(candidate_report(result)))


def evaluation_result(result):
    return json_value(asdict(evaluation_report(result)))


def from_json(kind, value):
    """Decode declared variants by their schema, never by trial construction or old formats."""
    if kind == Source:
        return source_from_json(value)
    origin, args = get_origin(kind), get_args(kind)
    if origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        from jsonschema import Draft202012Validator

        matches = [member for member in args if member is not type(None)
                   and Draft202012Validator(schema_for(member)).is_valid(value)]
        if len(matches) != 1:
            raise ValueError('wire value must match exactly one declared union variant')
        return from_json(matches[0], value)
    if is_dataclass(kind):
        return kind(**{name: from_json(annotation, value[name])
                       for name, annotation in get_type_hints(kind).items()})
    if origin is tuple:
        return tuple(from_json(args[0], item) for item in value) if args[-1] is Ellipsis else tuple(
            from_json(annotation, item) for annotation, item in zip(args, value))
    if origin is Mapping:
        return {key: from_json(args[1], item) for key, item in value.items()}
    if kind is bytes:
        return base64.b64decode(value, validate=True)
    if kind is float and value is None:
        return float('nan')
    return value


def describe(research):
    engine = research.pipeline
    return dict(protocol=PROTOCOL, context=asdict(research.context), precision=asdict(engine.context.precision),
        parameters=parameter_values(engine.settings),
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



def checkpoint_bytes(checkpoint: MiningCheckpoint) -> bytes:
    return json.dumps(dict(format=CHECKPOINT_FORMAT, checkpoint=json_value(asdict(checkpoint))),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def checkpoint_from_bytes(payload: bytes) -> MiningCheckpoint:
    record = json.loads(payload)
    if not isinstance(record, dict) or record.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('checkpoint protocol differs; start a new discovery run')
    from jsonschema import ValidationError, validate

    try:
        validate(record, dict(type='object', properties=dict(format=dict(const=CHECKPOINT_FORMAT),
            checkpoint=schema_for(MiningCheckpoint)), required=['format', 'checkpoint'], additionalProperties=False))
    except ValidationError as error:
        raise ValueError(error.message) from error
    return from_json(MiningCheckpoint, record['checkpoint'])


def telemetry_bytes(record: MiningStatus | MiningEpoch) -> bytes:
    return json.dumps(json_value(asdict(record)), ensure_ascii=False, allow_nan=False).encode('utf-8')


def encode_bundle(batch: ArrayBatch) -> bytes:
    if sum(array.nbytes for array in (batch.values, batch.observed_through, batch.fit_through)) > MAX_ARRAY_BYTES:
        raise ValueError('decoded array bundle exceeds the 1GiB limit')
    metadata = dict(format=BUNDLE, kind=batch.kind, ids=batch.ids, snapshot=batch.snapshot,
                    producer=batch.producer, dates=batch.dates, assets=batch.assets, complexity=batch.complexity)
    payload = encode_archive(metadata, batch.values, batch.observed_through, batch.fit_through)
    validate_bundle_size(payload)
    return payload


def validate_bundle_size(payload: bytes) -> None:
    """Inspect ZIP and NPY headers before NumPy can allocate declared array shapes."""
    if not 0 < len(payload) <= MAX_ARRAY_BYTES:
        raise ValueError('array bundle exceeds the 1GiB transport limit')
    try:
        with ZipFile(BytesIO(payload)) as archive:
            entries = archive.infolist()
            expected = {'metadata.npy', 'values.npy', 'observed_through.npy', 'fit_through.npy'}
            if len(entries) != len(expected) or {entry.filename for entry in entries} != expected:
                raise ValueError('array bundle fields differ from the declared format')
            if sum(entry.file_size for entry in entries) > MAX_ARRAY_BYTES:
                raise ValueError('decoded array bundle exceeds the 1GiB limit')
            for entry in entries:
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError('unsupported NPY header version in array bundle')
                    if dtype.hasobject or any(dimension < 0 for dimension in shape):
                        raise ValueError('array bundles require non-object arrays with nonnegative shapes')
                    declared = math.prod(shape) * dtype.itemsize
                    if declared > MAX_ARRAY_BYTES or stream.tell() + declared != entry.file_size:
                        raise ValueError('NPY shape differs from its bounded ZIP entry size')
    except (BadZipFile, EOFError) as error:
        raise ValueError('invalid NumPy array bundle') from error


def decode_bundle(payload: bytes) -> ArrayBatch:
    validate_bundle_size(payload)
    metadata, values, observed, fitted = decode_archive(payload)
    if metadata['format'] != BUNDLE:
        raise ValueError('batch protocol differs; export a current bundle')
    expected = {'format', 'kind', 'ids', 'snapshot', 'producer', 'dates', 'assets', 'complexity'}
    if set(metadata) != expected:
        raise ValueError('batch metadata fields differ from the declared format')
    return ArrayBatch(metadata['kind'], tuple(metadata['ids']), values, metadata['snapshot'],
                      metadata['producer'], tuple(metadata['dates']), tuple(metadata['assets']),
                      observed, fitted, None if metadata['complexity'] is None else tuple(metadata['complexity']))


def write_bundle(path: Path, batch: ArrayBatch) -> None:
    publish(path, encode_bundle(batch))


def load_bundle(path: Path) -> ArrayBatch:
    if path.stat().st_size > MAX_ARRAY_BYTES:
        raise ValueError('array bundle exceeds the 1GiB transport limit')
    return decode_bundle(path.read_bytes())

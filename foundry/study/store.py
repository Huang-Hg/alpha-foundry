"""Atomic publication, content-addressed arrays, CAS libraries and causal provenance."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from collections.abc import Callable
from foundry.types import MiningCheckpoint, MiningEpoch, MiningStatus
import hashlib
from io import BytesIO
import json
import re
import numpy as np
from foundry.types import ArrayBatch, ArraySource
from dataclasses import asdict
from foundry.types import Formula, EqualWeight
from foundry.portfolio.library import Library, Member, PROTOCOL
from foundry.types import ContextConflict
from foundry.numeric.semantics import BACKTEST_SEMANTIC_VERSION
from foundry.config import Settings
from foundry.numeric.semantics import MOMENTS_SEMANTIC_VERSION
from foundry.ops.registry import OPS_SEMANTIC_VERSION
from foundry.types import LoadedStudy, PrecisionPlan, StudySnapshot


def publish(path: Path, content: bytes) -> None:
    """Replace `path` atomically: a reader sees the old file or the new one, never a partial write."""
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


FORMAT = 'candidate-array-v2'


BUNDLE = 'candidate-batch-v2'


ARRAY_KEYS = {'metadata', 'values', 'observed_through', 'fit_through'}


def _json(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _encode(metadata, values, observed, fitted):
    output = BytesIO()
    np.savez(output, metadata=np.asarray(_json(metadata)), values=values,
             observed_through=observed, fit_through=fitted)
    return output.getvalue()


def _decode(payload):
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_KEYS:
            raise ValueError('array artifact fields differ from the declared format')
        return (json.loads(str(archive['metadata'].item())), archive['values'],
                archive['observed_through'], archive['fit_through'])


def _digest(metadata, arrays):
    digest = hashlib.sha256(_json(metadata).encode())
    for array in arrays:
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _records(batch):
    metadata = dict(format=FORMAT, kind=batch.kind, snapshot=batch.snapshot,
                    producer=batch.producer, dates=batch.dates, assets=batch.assets)
    complexities = (0.,) * len(batch.ids) if batch.complexity is None else batch.complexity
    for values, observed, fitted, complexity in zip(batch.values, batch.observed_through, batch.fit_through, complexities):
        metadata = dict(metadata, complexity=complexity)
        arrays = (np.ascontiguousarray(values), np.ascontiguousarray(observed, dtype=np.int64),
                  np.ascontiguousarray(fitted, dtype=np.int64))
        source = ArraySource(batch.kind, _digest(metadata, arrays), batch.snapshot, batch.producer, complexity)
        yield source, _encode(metadata, *arrays)


def _read(payload, source):
    metadata, values, observed, fitted = _decode(payload)
    if metadata['format'] != FORMAT:
        raise ValueError('array artifact protocol differs; register a current bundle')
    actual = ArraySource(metadata['kind'], _digest(metadata, (values, observed, fitted)),
                         metadata['snapshot'], metadata['producer'], metadata['complexity'])
    if actual != source:
        raise ValueError('array artifact digest or provenance differs from its source')
    for array in (values, observed, fitted):
        array.setflags(write=False)
    return ArrayBatch(source.kind, (source.digest,), values[None], source.snapshot, source.producer,
                      tuple(metadata['dates']), tuple(metadata['assets']), observed[None], fitted[None],
                      (metadata['complexity'],))


class FileArrayStore:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, source: ArraySource):
        if re.fullmatch('[0-9a-f]{64}', source.digest) is None:
            raise ValueError('array digest must be a SHA-256 hex value')
        return self.root / (source.digest + '.npz')

    def put(self, batch: ArrayBatch):
        self.root.mkdir(parents=True, exist_ok=True)
        sources = []
        for source, payload in _records(batch):
            path = self.path_for(source)
            if path.exists():
                self.read(source)
            else:
                publish(path, payload)
            sources.append(source)
        return tuple(sources)

    def read(self, source: ArraySource):
        return _read(self.path_for(source).read_bytes(), source)


class MemoryArrayStore:
    def __init__(self):
        self.artifacts = {}

    def put(self, batch: ArrayBatch):
        sources = []
        for source, payload in _records(batch):
            self.artifacts[source.digest] = payload
            sources.append(source)
        return tuple(sources)

    def read(self, source: ArraySource):
        return _read(self.artifacts[source.digest], source)


def write_bundle(path: Path, batch: ArrayBatch):
    """Export a host batch for RPC registration; device callers copy at this IO boundary."""
    publish(path, encode_bundle(batch))


def encode_bundle(batch: ArrayBatch) -> bytes:
    metadata = dict(format=BUNDLE, kind=batch.kind, ids=batch.ids, snapshot=batch.snapshot,
                    producer=batch.producer, dates=batch.dates, assets=batch.assets, complexity=batch.complexity)
    return _encode(metadata, batch.values, batch.observed_through, batch.fit_through)


def load_bundle(path: Path) -> ArrayBatch:
    """Read an external bundle without pickle or code deserialization."""
    return decode_bundle(path.read_bytes())


def decode_bundle(payload: bytes) -> ArrayBatch:
    metadata, values, observed, fitted = _decode(payload)
    if metadata['format'] != BUNDLE:
        raise ValueError('batch protocol differs; export a current bundle')
    expected = {'format', 'kind', 'ids', 'snapshot', 'producer', 'dates', 'assets', 'complexity'}
    if set(metadata) != expected:
        raise ValueError('batch metadata fields differ from the declared format')
    return ArrayBatch(metadata['kind'], tuple(metadata['ids']), values, metadata['snapshot'],
                      metadata['producer'], tuple(metadata['dates']), tuple(metadata['assets']),
                      observed, fitted, None if metadata['complexity'] is None else tuple(metadata['complexity']))


def member_from_json(record):
    return Member(source_from_json(record['source']), record['weight'])


class FileLibraryStore:
    persistent = True

    def __init__(self, path: Path):
        self.path = path

    def read(self, study):
        if not self.path.exists():
            return Library((), study)
        record = json.loads(self.path.read_bytes())
        if record['protocol'] != PROTOCOL:
            raise ValueError('library protocol differs; rebuild the library')
        if record['study'] != asdict(study):
            raise ContextConflict('library belongs to a different study')
        return Library(tuple(member_from_json(m) for m in record['members']), study)

    def commit(self, expected, replacement):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix('.lock')
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError as exc:
            raise ContextConflict('portfolio publication is already in progress') from exc
        try:
            if self.read(expected.study).version != expected.version:
                raise ContextConflict('portfolio changed on disk')
            payload = dict(protocol=PROTOCOL, study=asdict(replacement.study), members=[asdict(m) for m in replacement.members])
            publish(self.path, json.dumps(payload, ensure_ascii=False, indent=1).encode())
        finally:
            lock.unlink()


class MemoryLibraryStore:
    persistent = False

    def __init__(self, library):
        self.library = library

    def read(self, study):
        if study != self.library.study:
            raise ContextConflict('library belongs to a different study')
        return self.library

    def commit(self, expected, replacement):
        if self.library.version != expected.version:
            raise ContextConflict('portfolio changed')
        self.library = replacement


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def implementation_digest() -> str:
    """Bump explicit versions when mathematical or execution semantics change."""
    return digest((OPS_SEMANTIC_VERSION, BACKTEST_SEMANTIC_VERSION, MOMENTS_SEMANTIC_VERSION))


def study_snapshot(loaded: LoadedStudy, settings: Settings, precision: PrecisionPlan) -> StudySnapshot:
    admission = asdict(settings.portfolio)
    admission.pop('artifact_dir')
    configuration = digest(dict(operators=asdict(settings.expr), admission=admission,
        precision=precision.effective, return_atol=settings.cuda.return_atol,
        perpetual_host_cells=settings.cuda.perpetual_host_cells))
    values = dict(study=loaded.study, data_snapshot=loaded.panel.snapshot,
                  execution_snapshot=loaded.host_spec.snapshot, implementation=implementation_digest(),
                  configuration=configuration, data_contract=loaded.data_contract,
                  evaluation='in_sample_research', decision=loaded.market.decision_clock,
                  execution=loaded.market.execution_clock, allocation=loaded.market.allocation, precision=precision)
    identity = digest(dict(data=values['data_snapshot'], execution=values['execution_snapshot'],
        semantics=values['implementation'], configuration=configuration, window=asdict(loaded.study),
        allocation=asdict(loaded.market.allocation), decision=values['decision'], clock=values['execution']))
    return StudySnapshot(id=identity, **values)


def source_from_json(record):
    factory = {'formula': Formula, 'signals': ArraySource, 'target_weights': ArraySource,
               'target_exposure': ArraySource, 'equal_weight': EqualWeight}[record['kind']]
    return factory(**record)


class FileCheckpointStore:
    """One atomically published checkpoint with an injected wire codec."""

    def __init__(self, path: Path, encode: Callable[[MiningCheckpoint], bytes],
                 decode: Callable[[bytes], MiningCheckpoint]):
        self.path, self.encode, self.decode = path, encode, decode

    def read(self) -> MiningCheckpoint | None:
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            return None
        return self.decode(payload)

    def publish(self, checkpoint: MiningCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        publish(self.path, self.encode(checkpoint))


class FileTelemetrySink:
    """Publish immutable status snapshots and append completed epoch evidence."""

    def __init__(self, directory: Path, live: Path,
                 encode: Callable[[MiningStatus | MiningEpoch], bytes]):
        self.directory, self.live, self.encode = directory, live, encode
        directory.mkdir(parents=True, exist_ok=True)
        live.parent.mkdir(parents=True, exist_ok=True)

    def stopping(self) -> bool:
        return (self.directory / 'STOP').exists()

    def publish(self, status: MiningStatus) -> None:
        payload = self.encode(status)
        publish(self.directory / 'status.json', payload)
        publish(self.live, payload)

    def record(self, epoch: MiningEpoch) -> None:
        with (self.directory / 'epochs.jsonl').open('ab') as stream:
            stream.write(self.encode(epoch) + b'\n')


class RunLease:
    """Exclusive local ownership of one discovery output directory."""

    def __init__(self, directory: Path):
        self.directory, self.lock = directory, directory / '.running'

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()))
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.lock.unlink()

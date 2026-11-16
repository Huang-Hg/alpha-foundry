"""Atomic publication, content-addressed arrays, CAS libraries and causal provenance."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
from foundry.study.identity import FORMAT, canonical_json, digest_for, host_leaves, metadata_for, source_for


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


ARRAY_KEYS = {'metadata', 'values', 'observed_at', 'fit_at', 'update_mask'}


def encode_archive(metadata, values, observed, fitted, update_mask=None):
    output = BytesIO()
    np.savez(output, metadata=np.asarray(canonical_json(metadata)), values=values,
             observed_at=observed, fit_at=fitted,
             update_mask=np.empty(0, dtype=bool) if update_mask is None else update_mask)
    return output.getvalue()


def decode_archive(payload):
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_KEYS:
            raise ValueError('array artifact fields differ from the declared format')
        metadata = json.loads(str(archive['metadata'].item()))
        if not isinstance(metadata, dict):
            raise ValueError('array metadata must be a JSON object')
        mask = archive['update_mask']
        if mask.dtype != np.dtype(bool):
            raise ValueError('array update_mask must have boolean dtype')
        return (metadata, archive['values'],
                archive['observed_at'], archive['fit_at'], None if mask.shape == (0,) else mask)


def _payload(batch: ArrayBatch, index: int):
    metadata = metadata_for(batch, index)
    arrays = (batch.values[index], batch.observed_at[index], batch.fit_at[index])
    if batch.update_mask is not None:
        arrays += (batch.update_mask[index],)
    return metadata, arrays


def publish_archive(path: Path, metadata: dict, arrays: tuple[np.ndarray, ...]) -> None:
    """Stream one NPZ to its atomic staging file, without a payload-sized BytesIO."""
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, metadata=np.asarray(canonical_json(metadata)), values=arrays[0],
                     observed_at=arrays[1], fit_at=arrays[2],
                     update_mask=arrays[3] if len(arrays) == 4 else np.empty(0, dtype=bool))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read(payload, source):
    metadata, values, observed, fitted, mask = decode_archive(payload)
    expected = {'format', 'kind', 'unit', 'snapshot', 'producer', 'decision_times',
                'assets', 'update_mask', 'complexity'}
    if set(metadata) != expected:
        raise ValueError('array artifact metadata fields differ from the declared format')
    if metadata['format'] != FORMAT:
        raise ValueError('array artifact protocol differs; register a current bundle')
    if metadata['update_mask'] != ('all' if mask is None else 'explicit'):
        raise ValueError('array artifact update mask differs from its metadata')
    arrays = (values, observed, fitted) + (() if mask is None else (mask,))
    actual = source_for(metadata, digest_for(metadata, arrays, tuple(host_leaves(a[None])[0] for a in arrays)))
    if actual != source:
        raise ValueError('array artifact digest or provenance differs from its source')
    for array in arrays:
        array.setflags(write=False)
    return ArrayBatch(source.kind, (source.digest,), values[None], source.snapshot, source.producer,
                      tuple(metadata['decision_times']), tuple(metadata['assets']), observed[None], fitted[None],
                      (metadata['complexity'],), source.unit, None if mask is None else mask[None])


class FileArrayStore:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, source: ArraySource):
        if source.scope:
            raise ValueError('resident source must be retained before file access')
        if re.fullmatch('[0-9a-f]{64}', source.digest) is None:
            raise ValueError('array digest must be a SHA-256 hex value')
        return self.root / (source.digest + '.npz')

    def put(self, batch: ArrayBatch, sources: tuple[ArraySource, ...]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        def write(index: int) -> None:
            source = sources[index]
            path = self.path_for(source)
            if path.exists():
                self.read(source)
            else:
                metadata, arrays = _payload(batch, index)
                publish_archive(path, metadata, arrays)

        if len(sources) == 1:
            write(0)
        else:
            # IO workers share the bounded immutable host batch.
            with ThreadPoolExecutor(max_workers=min(4, len(sources))) as workers:
                for _ in workers.map(write, range(len(sources))):
                    pass

    def read(self, source: ArraySource):
        return _read(self.path_for(source).read_bytes(), source)


class MemoryArrayStore:
    def __init__(self):
        self.artifacts = {}

    def put(self, batch: ArrayBatch, sources: tuple[ArraySource, ...]) -> None:
        for index, source in enumerate(sources):
            metadata, arrays = _payload(batch, index)
            self.artifacts[source.digest] = encode_archive(metadata, *arrays)

    def read(self, source: ArraySource):
        return _read(self.artifacts[source.digest], source)


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
        if any(isinstance(m.source, ArraySource) and m.source.scope for m in replacement.members):
            raise ValueError('portfolio publication requires retained array sources')
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
        if any(isinstance(m.source, ArraySource) and m.source.scope for m in replacement.members):
            raise ValueError('portfolio publication requires retained array sources')
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
        clock=asdict(settings.clock), sizing=asdict(settings.sizing), account=asdict(settings.account),
        execution=asdict(settings.execution), decision_times=loaded.window.clock.decision_times,
        calendar=loaded.window.clock.calendar, clock_provenance=loaded.window.clock.provenance,
        precision=precision.effective, return_atol=settings.cuda.return_atol,
        perpetual_host_cells=settings.cuda.perpetual_host_cells))
    values = dict(study=loaded.study, data_snapshot=loaded.panel.snapshot,
                  execution_snapshot=loaded.host_spec.snapshot, implementation=implementation_digest(),
                  configuration=configuration, data_contract=loaded.data_contract,
                  evaluation='in_sample_research', decision=loaded.market.decision_clock,
                  execution=loaded.market.execution_clock, allocation=loaded.market.allocation, precision=precision)
    identity = digest(dict(data=values['data_snapshot'], execution=values['execution_snapshot'],
        program_execution=None if loaded.program_spec is None else loaded.program_spec.snapshot,
        semantics=values['implementation'], configuration=configuration, window=asdict(loaded.study),
        allocation=asdict(loaded.market.allocation), decision=values['decision'], clock=values['execution']))
    return StudySnapshot(id=identity, **values)


def source_from_json(record):
    factory = {'formula': Formula, 'scores': ArraySource, 'targets': ArraySource,
               'equal_weight': EqualWeight}[record['kind']]
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

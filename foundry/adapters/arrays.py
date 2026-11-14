"""NumPy bundles and immutable content-addressed external candidate artifacts."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np

from foundry.adapters.files import publish
from foundry.candidates import ArrayBatch, ArraySource


FORMAT = 'candidate-array-v1'
BUNDLE = 'candidate-batch-v1'
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
    for values, observed, fitted in zip(batch.values, batch.observed_through, batch.fit_through):
        arrays = (np.ascontiguousarray(values), np.ascontiguousarray(observed, dtype=np.int64),
                  np.ascontiguousarray(fitted, dtype=np.int64))
        source = ArraySource(batch.kind, _digest(metadata, arrays), batch.snapshot, batch.producer)
        yield source, _encode(metadata, *arrays)


def _read(payload, source):
    metadata, values, observed, fitted = _decode(payload)
    if metadata['format'] != FORMAT:
        raise ValueError('array artifact protocol differs; register a current bundle')
    actual = ArraySource(metadata['kind'], _digest(metadata, (values, observed, fitted)),
                         metadata['snapshot'], metadata['producer'])
    if actual != source:
        raise ValueError('array artifact digest or provenance differs from its source')
    for array in (values, observed, fitted):
        array.setflags(write=False)
    return ArrayBatch(source.kind, (source.digest,), values[None], source.snapshot, source.producer,
                      tuple(metadata['dates']), tuple(metadata['assets']), observed[None], fitted[None])


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
    metadata = dict(format=BUNDLE, kind=batch.kind, ids=batch.ids, snapshot=batch.snapshot,
                    producer=batch.producer, dates=batch.dates, assets=batch.assets)
    publish(path, _encode(metadata, batch.values, batch.observed_through, batch.fit_through))


def load_bundle(path: Path) -> ArrayBatch:
    """Read an external bundle without pickle or code deserialization."""
    metadata, values, observed, fitted = _decode(path.read_bytes())
    if metadata['format'] != BUNDLE:
        raise ValueError('batch protocol differs; export a current bundle')
    expected = {'format', 'kind', 'ids', 'snapshot', 'producer', 'dates', 'assets'}
    if set(metadata) != expected:
        raise ValueError('batch metadata fields differ from the declared format')
    return ArrayBatch(metadata['kind'], tuple(metadata['ids']), values, metadata['snapshot'],
                      metadata['producer'], tuple(metadata['dates']), tuple(metadata['assets']),
                      observed, fitted)

"""Content identities from typed metadata and ordered SHA-256 chunk digests.

Only leaf digests cross the device boundary. This is a versioned tree hash,
not the SHA-256 of concatenated array bytes and not a FIPS certification claim.
"""
import hashlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from foundry.types import Array, ArrayBatch, ArraySource, HostArray

FORMAT = 'candidate-array-v3'
CHUNK_BYTES = 4096


def canonical_json(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def host_leaves(values: HostArray) -> HostArray:
    rows = np.ascontiguousarray(values).view(np.uint8).reshape(len(values), -1)
    leaves = max(1, (rows.shape[1] + CHUNK_BYTES - 1) // CHUNK_BYTES)
    result = np.empty((len(rows), leaves, 32), dtype=np.uint8)
    for i, row in enumerate(rows):
        view = memoryview(row)
        for j in range(leaves):
            result[i, j] = np.frombuffer(hashlib.sha256(view[j*CHUNK_BYTES:(j+1)*CHUNK_BYTES]).digest(), dtype=np.uint8)
    return result


class DeviceHasher:
    """Explicitly initialized CUDA SHA-256 leaves; no import-time device work."""
    def __init__(self, cp: ModuleType):
        self.cp = cp
        self.kernel = cp.RawKernel((Path(__file__).parent/'src/array_hash.cu').read_text(), 'sha256_leaves')

    def leaves(self, values: Array) -> HostArray:
        cp = self.cp
        if isinstance(values, np.ndarray):
            return host_leaves(values)
        contiguous = cp.ascontiguousarray(values)
        row_bytes = contiguous.nbytes // len(contiguous)
        leaves = max(1, (row_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES)
        output = cp.empty((len(contiguous), leaves, 32), dtype=cp.uint8)
        self.kernel(((len(contiguous)*leaves+127)//128,), (128,),
                    (contiguous, output, np.int64(row_bytes), np.int32(leaves), np.int32(len(contiguous))))
        return cp.asnumpy(output)


def metadata_for(batch: ArrayBatch, index: int) -> dict:
    return dict(format=FORMAT, kind=batch.kind, snapshot=batch.snapshot, producer=batch.producer,
                dates=batch.dates, assets=batch.assets,
                complexity=0. if batch.complexity is None else batch.complexity[index])


def digest_for(metadata: dict, arrays: tuple[Array, ...], leaves: tuple[HostArray, ...]) -> str:
    root = hashlib.sha256(canonical_json(metadata).encode())
    for array, hashes in zip(arrays, leaves):
        root.update(str((array.dtype.str, array.shape)).encode())
        root.update(memoryview(hashes).cast('B'))
    return root.hexdigest()


def source_for(metadata: dict, digest: str, scope: str = '') -> ArraySource:
    return ArraySource(metadata['kind'], digest, metadata['snapshot'], metadata['producer'], metadata['complexity'], scope)

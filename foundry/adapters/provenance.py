"""Source and configuration fingerprints, read once at explicit study startup."""
from dataclasses import asdict
import hashlib
import json

from foundry.config import ROOT
from foundry.contracts import StudySnapshot


def implementation_digest():
    digest = hashlib.sha256()
    for source in sorted((ROOT / 'foundry').rglob('*')):
        if source.suffix in ('.py', '.cpp', '.cu', '.cuh', '.hpp', '.h'):
            digest.update(source.relative_to(ROOT).as_posix().encode())
            digest.update(source.read_bytes())
    digest.update(b'build.sh')
    digest.update((ROOT / 'build.sh').read_bytes())
    return digest.hexdigest()


def study_snapshot(loaded, settings, precision):
    configuration = hashlib.sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest()
    values = dict(study=loaded.study, data_snapshot=loaded.panel.snapshot,
                  execution_snapshot=loaded.host_spec.snapshot, implementation=implementation_digest(),
                  configuration=configuration, data_contract=loaded.data_contract,
                  evaluation='in_sample_research', decision=loaded.market.decision_clock,
                  execution=loaded.market.execution_clock, allocation=loaded.market.allocation, precision=precision)
    serial = dict(values, study=asdict(loaded.study), data_contract=asdict(loaded.data_contract),
                  allocation=asdict(loaded.market.allocation), precision=asdict(precision))
    identity = hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
    return StudySnapshot(id=identity, **values)

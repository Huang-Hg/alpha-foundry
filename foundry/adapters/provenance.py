"""Causal snapshot identity; presentation and transport code are not inputs."""
from dataclasses import asdict
import hashlib
import json

from foundry.backtest.spec import BACKTEST_SEMANTIC_VERSION
from foundry.config import Settings
from foundry.numeric.semantics import MOMENTS_SEMANTIC_VERSION
from foundry.ops.registry import OPS_SEMANTIC_VERSION
from foundry.types import LoadedStudy, PrecisionPlan, StudySnapshot


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

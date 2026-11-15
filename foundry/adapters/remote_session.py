"""Portable session transport; host arrays upload without remote path plumbing."""
from dataclasses import asdict

import numpy as np

from foundry.adapters.arrays import encode_bundle
from foundry.adapters.serialization import from_json
from foundry.types import FamilyResult
from foundry.types import (CommitReport, DataGrid, EvaluationReport, PortfolioReport,
                                   RegisteredBatch, ResearchContext, WeightCommit, WeightReport)
from foundry.service import client


class RemoteBackend:
    def __init__(self, address):
        self.endpoint = client.info() if address is None else dict(address=address[0], port=address[1])

    def _call(self, method, parameters, kind, payload=b''):
        return from_json(kind, client.call(method, parameters, endpoint=self.endpoint, payload=payload))

    def context(self):
        self.description = self.describe()
        return from_json(ResearchContext, self.description['context'])

    def refresh(self):
        return self.context()

    def grid(self):
        return from_json(DataGrid, self.description['axes'])

    def describe(self):
        return client.call('research.describe', {}, endpoint=self.endpoint)

    def evaluate(self, candidates, context):
        return self._call('research.evaluate', dict(context=context, candidates=[asdict(c) for c in candidates]), EvaluationReport)

    def evaluate_device(self, candidates, context):
        raise ValueError('resident device results require foundry.open(); remote sessions return structured reports')

    def commit(self, candidates, context):
        return self._call('research.commit', dict(context=context, candidates=[asdict(c) for c in candidates]), CommitReport)

    def register(self, batch, context):
        payload = encode_bundle(batch)
        return self._call('research.register_batch', dict(context=context, bytes=len(payload)), RegisteredBatch, payload)

    def explore(self, seeds, context):
        return self._call('research.family', dict(context=context, seeds=[asdict(s) for s in seeds]), FamilyResult)

    def portfolio(self, context):
        return self._call('research.portfolio', dict(context=context), PortfolioReport)

    def trial_weights(self, weights, context):
        return self._call('research.trial_weights', dict(context=context, weights=np.asarray(weights).tolist()), WeightReport)

    def commit_weights(self, weights, context):
        return self._call('research.commit_weights', dict(context=context, weights=list(weights)), WeightCommit)

    def close(self):
        """Requests own their sockets; the connected compute server remains running."""

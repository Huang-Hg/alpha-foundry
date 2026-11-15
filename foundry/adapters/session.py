"""In-process binding of the public session to the shared research use cases."""
from foundry.adapters.serialization import describe, evaluation_report
from foundry.research import Research
from foundry.types import DataGrid, RegisteredBatch


class LocalBackend:
    def __init__(self, runtime, research):
        self.runtime, self.research = runtime, research

    def context(self):
        return self.research.context

    def refresh(self):
        with self.runtime.resources.scope():
            previous = self.research
            self.research = Research(self.runtime.engine, previous.store, previous.arrays, previous.portfolio.null_shift)
        return self.research.context

    def grid(self):
        engine = self.research.engine
        return DataGrid(engine.context.id, tuple(str(day) for day in engine.dates), tuple(engine.panel.assets))

    def describe(self):
        return describe(self.research)

    def evaluate(self, candidates, context):
        return evaluation_report(self.research.evaluate(candidates, context))

    def evaluate_device(self, candidates, context):
        return self.research.evaluate_device(tuple(c.source for c in candidates), context, report=True)

    def commit(self, candidates, context):
        return evaluation_report(self.research.commit(candidates, context))

    def register(self, batch, context):
        with self.runtime.resources.scope():
            candidates = self.research.register_batch(batch, context)
        return RegisteredBatch(self.research.context, candidates)

    def explore(self, seeds, context):
        return self.research.family(seeds, context)

    def portfolio(self, context):
        return self.research.portfolio_report(context)

    def trial_weights(self, weights, context):
        return self.research.evaluate_weights(weights, context)

    def commit_weights(self, weights, context):
        return self.research.commit_weights(weights, context)

    def close(self):
        self.runtime.close()

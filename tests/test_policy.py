"""Native search policy (foundry.search.policy): grammar, cost and causality of samples, trajectory-balance training,
paired teachers, checkpoints and the CEM weight sampler. Needs neither a GPU nor market data."""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from foundry.expr.parse import validate
from foundry.search.policy import CEM, Policy

FIELDS = ['x', 'y', 'z']
STEPS, LR, OPD = 4, .002, .1


def policy(fields, seed):
    return Policy(fields, seed, 13, 25, 2, (730, 365))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = policy(FIELDS, 25)

    def test_grammar_cost_and_causality(self):
        batch = self.policy.sample(256, 4, [0.] * 16)
        self.assertEqual(len(batch['expressions']), 256 + 12 * batch['probe_count'])
        for expr in batch['expressions']:
            v = validate(expr, FIELDS, 25)
            self.assertTrue(v['ok'], (expr, v))
            self.assertLessEqual(v['cost'], 25)
            self.assertLessEqual(len(re.findall(r'[A-Za-z_]\w*|-?\d+\.?\d*', expr)), 13, expr)

            def check(node):
                """Time-series operators stacked on the deepest path below (and including) node."""
                if node.leaf:
                    return 0
                return node.op.startswith('ts_') + max((check(child) for child in node.children), default=0)
            self.assertLessEqual(check(v['tree'].root), 2, expr)

    def test_tb_gradient_learning_and_stale_batch(self):
        batch = self.policy.sample(32, 0, [0.] * 16)
        rewards = [-6.] * 32
        self.assertLess(self.policy.gradient_check(rewards)['max_gradient_error'], 1e-5)
        learned = self.policy.train(rewards, batch['epoch'], STEPS, LR, 0)
        self.assertLess(learned['loss_after'], learned['loss_before'])
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            self.policy.train(rewards, batch['epoch'], STEPS, LR, OPD)

    def test_reliable_teacher_and_ties(self):
        batch = self.policy.sample(32, 2, [0.] * 16)
        self.assertGreater(batch['probe_count'], 0)
        rewards = [-6.] * 32 + [1., 3., -2.] * 4 * batch['probe_count']
        learned = self.policy.train(rewards, batch['epoch'], STEPS, LR, OPD)
        self.assertEqual(learned['teachers'], batch['probe_count'])
        batch = self.policy.sample(32, 2, [0.] * 16)
        learned = self.policy.train([-6.] * len(batch['expressions']), batch['epoch'], STEPS, LR, OPD)
        self.assertEqual(learned['teachers'], 0)

    def test_checkpoint_restores_rng_and_rejects_grammar_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.bin'
            self.policy.checkpoint(path)
            first = self.policy.sample(32, 1, [0.] * 16)
            self.policy.checkpoint(path, load=True)
            second = self.policy.sample(32, 1, [0.] * 16)
            self.assertEqual(first['expressions'], second['expressions'])
            with self.assertRaisesRegex(RuntimeError, 'grammar mismatch'):
                policy(['different'], 1).checkpoint(path, load=True)

    def test_cem_additive_weights(self):
        sampler = CEM([0., 0., 0.])
        data = sampler.sample(64, 25)
        self.assertEqual(data.shape, (64, 3))
        self.assertTrue(((data > 0) & (data < 1)).all())
        self.assertGreater(np.abs(data.sum(1) - 1).max(), .1)          # independent scales, not shares
        sampler.update(data[:, 0], 16)
        self.assertGreater(sampler.mean[0], sampler.mean[1])


if __name__ == '__main__':
    unittest.main()

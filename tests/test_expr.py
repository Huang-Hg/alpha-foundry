"""Formula layer: registry, parser, validator, tree hash, CPU evaluation. No data, no GPU."""
import unittest

import numpy as np

from foundry.expr.evaluate import backend, evaluate
from foundry.expr.parse import MAX_COST, ExprError, parse, validate
from foundry.ops.registry import ARITY, OPS

FIELDS = ('x', 'y', 'z')


class ParseTests(unittest.TestCase):
    def test_every_operator_round_trips_through_pretty(self):
        samples = {'unary': '{}(x)', 'binary': '{}(x, y)', 'binary_const': '{}(x, 0.5)', 'ternary': '{}(x, y, z)',
                   'ts': '{}(x, 20)', 'ts_param': '{}(x, 20, 0.5)', 'pair': '{}(x, y, 20)', 'pair_param': '{}(x, y, 20, 2)'}
        for name, op in OPS.items():
            with self.subTest(op=name):
                text = samples[op.kind].format(name)
                tree = parse(text, FIELDS)
                self.assertEqual(tree.pretty(), text)
                self.assertEqual(parse(tree.pretty(), FIELDS).hash, tree.hash)

    def test_commutative_operators_hash_alike(self):
        self.assertEqual(parse('add(x, ts_mean(y, 5))', FIELDS).hash, parse('add(ts_mean(y, 5), x)', FIELDS).hash)
        self.assertEqual(parse('ts_corr(x, y, 9)', FIELDS).hash, parse('ts_corr(y, x, 9)', FIELDS).hash)
        self.assertNotEqual(parse('sub(x, y)', FIELDS).hash, parse('sub(y, x)', FIELDS).hash)

    def test_errors_name_the_position_and_the_fix(self):
        for text, part in (('ts_mean(x)', '需要 2 个参数'), ('foo(x)', '未知算子'), ('ts_mean(q, 5)', '未知字段'),
                           ('ts_mean(x, 1)', '越界'), ('ts_mean(x, 2.5)', '必须是整数'), ('mul_const(x, 0)', '退化'),
                           ('add(x, y', '意外结束'), ('0.5', '裸常数'), ('sign(0.5)', '必须是子表达式')):
            with self.subTest(text=text):
                with self.assertRaises(ExprError) as e:
                    parse(text, FIELDS)
                self.assertIn(part, str(e.exception))

    def test_validator(self):
        ok = validate('gt(ts_mean(x, 20), 0.5)', FIELDS)
        self.assertTrue(ok['ok'])
        self.assertEqual((ok['fields'], ok['windows'], ok['n_ops']), (['x'], [20], 2))
        self.assertIn('非因果', validate('ts_rank(x, 32)', FIELDS)['error'])
        self.assertTrue(validate('ts_rank(x, 31)', FIELDS)['ok'])
        self.assertIn('纯常数', validate('add(1, 2)', FIELDS)['error'])
        deep = 'x'
        while parse(deep, FIELDS).cost() <= MAX_COST:
            deep = f'ts_mean({deep}, 5)'
        self.assertIn('cost', validate(deep, FIELDS)['error'])

    def test_scalars_are_continuous_and_bare_numbers_are_constant_panels(self):
        t = parse('gt(x, 0.37)', FIELDS)
        self.assertEqual([c.kind for c in t.root.children], ['operand', 'const_add'])
        self.assertEqual(parse('mul_const(x, 1e-6)', FIELDS).root.children[1].op, 1e-6)


class RegistryTests(unittest.TestCase):
    def test_cpu_backend_implements_the_registry(self):
        from foundry.ops.cpu import IMPL
        self.assertEqual(set(IMPL), set(OPS))
        self.assertEqual(set(ARITY), {op.kind for op in OPS.values()})

    def test_cpu_evaluation_shares_subtrees(self):
        rng = np.random.default_rng(1)
        fields = {k: rng.standard_normal((300, 7)) for k in FIELDS}
        cache = {}
        a = evaluate(parse('add(ts_mean(x, 5), ts_mean(x, 5))', FIELDS).root, fields, backend('cpu'), cache)
        np.testing.assert_array_equal(a, 2 * evaluate(parse('ts_mean(x, 5)', FIELDS).root, fields, backend('cpu'), {}))
        self.assertEqual(len(cache), 2)                     # ts_mean once, add once
        self.assertTrue(np.isnan(a[:4]).all() and np.isfinite(a[4:]).all())


if __name__ == '__main__':
    unittest.main()

"""Human-readable inspection of the assembled market and operator registry."""
import numpy as np
from foundry.ops.registry import ARITY, MAX_WINDOW, MIN_WINDOW, OPS


def fields(runtime):
    engine, market, panel = runtime.engine, runtime.market, runtime.panel
    universe = engine.to_host(engine.universe)[engine.start:]
    print(f'{market.name}: {len(panel.vocabulary)} 个字段\n')
    for category, names in panel.categories().items():
        scope = panel.fields[names[0]].scope
        print(f'{category} [{scope}] {market.categories[category]}')
        for name in names:
            values = engine.to_host(panel[name])[engine.start:][universe]
            finite = values[np.isfinite(values)]
            if len(finite):
                p1, p50, p99 = np.percentile(finite, (1, 50, 99))
                print(f'  {name:20s} 覆盖 {len(finite) / len(values):.3f} p1={p1:.4g} p50={p50:.4g} p99={p99:.4g}')
            else:
                print(f'  {name:20s} 无可见历史')


def ops(expr_config):
    for n in sorted(set(ARITY.values())):
        print(f'{n} 参数: ' + ' '.join(sorted(k for k, op in OPS.items() if ARITY[op.kind] == n)))
    print(f'整数窗口 {MIN_WINDOW}..{MAX_WINDOW}; 所有算子使用当前及历史数据; 标量参数连续取值')
    print(f'cost 上限 {expr_config.max_cost} ([expr] max_cost)')
    print('表达式输出 clip[0,1] 后除以当日可持有资产数; NaN 或不符合资格的资产权重为零。')

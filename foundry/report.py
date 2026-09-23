"""Terminal reports of the formula commands (eval / family / fields / ops)."""
from __future__ import annotations

import collections

import numpy as np

from foundry.ops.registry import ARITY, MAX_WINDOW, MIN_WINDOW, OPS


def rows(rows, quiet=False):
    """One block per formula; quiet prints only the admitted ones and a count per stage."""
    for row in rows:
        if row['passed'] or not quiet:
            _row(row, not quiet)
    if quiet:
        t = collections.Counter(r['stopped_at'] or 'PASS' for r in rows)
        names = {'G1': '结构/因果', 'G2': '组合 Sharpe / 收益', 'PASS': '全过'}
        print('  死因分布  ' + '   '.join(f'{names[k]} {t[k]:,}' for k in ('G1', 'G2', 'PASS') if t[k]))
    print(f'\n汇总:{len(rows)} 条,{sum(bool(r["passed"]) for r in rows)} 条通过组合准入')


def _row(row, verbose):
    print(f'{"✅" if row["passed"] else "❌"} {row["expr"]}')
    if not row['ok']:
        print(f'    {row["error"]}')
        return
    print(f'    sharpe {row["sharpe"]:+.4f}  CAGR {row["cagr"]:+.1%}  turnover {row["turnover"]:.4f}  '
          f'exposure {row["exposure"]:.3f}  maxDD {row["maxdd"]:+.3f}  margin {row["margin_bps"]:.0f}bps  年化 {row["ann_ret"]:+.1%}')
    print(f'    {row["explain"]}')
    c = row.get('combo')
    if c:
        print(f'    当前组合 K={c["n"]} · 候选与组合 ρ={c["rho"]:.3f}(仅诊断)')
        if 'sharpe_after' in c:
            label = '可行权重' if c['feasible'] else '约束违反最小的试探点(未通过)'
            print(f'    {label} {c["w"]:.3f}:近期 Sharpe {c["sharpe"]:.3f} → {c["sharpe_after"]:.3f}; '
                  f'全期 Sharpe {c["full_sharpe"]:.3f} → {c["full_sharpe_after"]:.3f}; '
                  f'CAGR {c["cagr"]:.1%} → {c["cagr_after"]:.1%}; 回撤 {c["maxdd"]:.1%} → {c["maxdd_after"]:.1%}')
    if verbose and row.get('yearly'):
        print('    逐年 Sharpe ' + ' '.join(f'{y["year"]}:{y["sharpe"]:+.2f}' for y in row['yearly']))
    if row.get('accepted'):
        print(f'    → 已按权重 {row["weight"]:.3f} 并入组合库')


def family(res):
    def num(x, spec='+.3f'):
        return '—' if x is None else format(x, spec)

    def delta(r):
        return f'组合 ΔSharpe {num(r["gain"])}  ΔCAGR {num(r["d_cagr"], "+.1%")}  Δ回撤 {num(r["d_maxdd"], "+.1%")}'

    def spread(r):
        return (f'ΔSharpe 中位 {num(r["median"])}(p25 {num(r["p25"])}, p75 {num(r["p75"])})  '
                f'ΔCAGR 中位 {num(r["median_cagr"], "+.1%")}  Δ回撤 中位 {num(r["median_maxdd"], "+.1%")}')

    marks = {'成立': '✅', '稀疏': '⚠️', '尖峰': '⚠️', '无效': '❌'}
    print('Δ = 加入后组合 − 加入前组合;Sharpe 为主维度;Δ回撤 > 0 = 回撤变浅')
    for f in res['families']:
        s, g, t, c = f['seed'], f['grid'], f['twins'], f['centre']
        print(f'{marks[f["verdict"]]} {f["verdict"]}  {f["hypothesis"]}')
        print(f'    种子  {"过" if s["passed"] else "未过"}  {delta(s)}  自身 Sharpe {num(s["sharpe"], ".2f")}  '
              f'持仓日 {num(s["attend"], ".1%")}  ρ {num(s["rho"], ".2f")}  在自己邻域里排 {num(f["seed_rank"], ".0%")}   {s["expr"]}')
        print(f'    邻域  {g["n"]:4d} 点  通过 {num(g["pass_rate"], ".1%")}  {spread(g)}')
        print(f'    孪生  {t["n"]:4d} 点  通过 {num(t["pass_rate"], ".1%")}  {spread(t)}   ← 同样结构、信号推迟 1-3 年:噪声下的水平')
        if c:
            print(f'    中心  一格邻居 ΔSharpe 中位 {num(c["neighbourhood_median"])}  {"过" if c["passed"] else "未过"}  {delta(c)}  '
                  f'持仓日 {num(c["attend"], ".1%")}   {c["expr"]}')
        for name, x in f['fields'].items():
            print(f'    字段  {name} → 换成同类其余 {x["substitutes"]} 个:ΔSharpe 中位 {num(x["median_gain"])},通过 {x["passed"]}')
    for s in res['skipped']:
        print(f'❌ 未展开  {s["expr"]}\n    {s["error"]}')
    v = [f['verdict'] for f in res['families']]
    print(f'\n汇总:{len(v)} 条假设公式,成立 {v.count("成立")},稀疏 {v.count("稀疏")},尖峰 {v.count("尖峰")},'
          f'无效 {v.count("无效")};本轮共评价 {res["variants"]:,} 个变体(多重检验计数)')


def fields():
    """The formula vocabulary of the active market, measured on its research window (universe cells only)."""
    from foundry.config import MarketConfig
    from foundry.expr.evaluate import backend
    from foundry.market.base import market
    m = market(MarketConfig().name)
    panel, _, window = m.load(backend('cpu'))
    universe = m.universe(panel)[window.start:]
    print(f'{m.name}:{len(panel.vocabulary)} 个字段(价格与成交量不入词表,仅供回测定价)\n')
    for cat, names in panel.categories().items():
        scope = panel.fields[names[0]].scope
        print(f'### {cat}  [{scope}]  {m.categories.get(cat, "")}')
        if scope == 'market':
            print('    ⚠ 市场级:跨资产恒等 ⇒ 只能表达择时,不能表达选币')
        for f in names:
            x = panel[f]
            rows_with_data = np.flatnonzero(np.isfinite(x).any(1))
            v = x[window.start:][universe]
            fin = v[np.isfinite(v)]
            p = np.percentile(fin, (0, 1, 50, 99, 100)) if len(fin) else [np.nan] * 5
            unit = p[0] >= 0 and p[4] <= 1
            rng = '天然 ∈[0,1] 可直接作敞口' if unit else f'p1={p[1]:.4g} p50={p[2]:.4g} p99={p[3]:.4g}'
            warn = '  ⚠ 宽域水平量,clip[0,1] 后近乎恒满仓' if not unit and abs(p[3]) > 100 else ''
            print(f'    {f:14s} 覆盖 {len(fin) / max(len(v), 1):.3f}  首行 {rows_with_data[0] if len(rows_with_data) else "—":<5} {rng}{warn}')
        print()


def ops():
    from foundry.expr.parse import MAX_COST
    for n in sorted(set(ARITY.values())):
        print(f'{n} 参数: ' + ' '.join(sorted(k for k, op in OPS.items() if ARITY[op.kind] == n)))
    causal = ', '.join(f'{k} ≤ {op.causal_window}' for k, op in OPS.items() if op.causal_window)
    print(f'\n整数窗口 {MIN_WINDOW}..{MAX_WINDOW};因果窗口上限:{causal};标量参数连续取值')
    print(f'cost 上限 {MAX_COST}([expr] max_cost)')
    print('\n敞口语义:表达式输出 → clip[0,1] = 每资产敞口 → 权重 = 敞口 / 当日可持有资产数。'
          '\n          NaN 或不在当日可持有集合 → 0(现金)。符号是一等参数:负值全被 clip 掉,方向写反 = 空仓。')

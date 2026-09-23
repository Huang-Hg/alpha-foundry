"""Q25 alpha 运行器 —— 子 agent 的唯一入口。

子 agent 只做两件事:**写 alpha、跑 run.py 看结果**。数据、口径、门、因果校验全在基础设施里。

    # 看有哪些字段 / 算子
    python tmp/q25_funnel/run.py --fields
    python tmp/q25_funnel/run.py --ops

    # FastExpr 车道:直接给表达式(可多条)
    python tmp/q25_funnel/run.py "sign(ts_slope(btc_feepertx, 365))" "ts_chanpos(btc_fees, 120)"
    python tmp/q25_funnel/run.py --file my_exprs.txt          # 一行一条,# 开头为注释

    # Python 车道:文件里定义 alpha(data, liquid) -> (T,S) 敞口
    python tmp/q25_funnel/run.py --python my_alpha.py

    # 假设族:每条公式在 GPU 上展开成窗口/阈值网格 + 同类字段替换 + 噪声孪生,判整片邻域(family.py)
    python tmp/q25_funnel/run.py --family hyp.txt --json out.json   # `# 假设` 注释行管到下一条注释为止

    # 产物
    python tmp/q25_funnel/run.py --file e.txt --json out.json   # 机读结果
    python tmp/q25_funnel/run.py --file e.txt --accept          # 全过者按最优权并入组合库(主控专用)

候选直接参与当前组合优化:权重网格上的实际合成持仓扣成本回测。组合近期 Sharpe 提升、全期 Sharpe 不下降,
且全期/近期收益的相对变化不小于回撤深度的相对变化才接受;单条年度表现和相关仅作诊断。
每次 --accept 使用已精算通过的权重,下一条对更新后的组合判定。空库从等权基准开始。

    python tmp/q25_funnel/run.py --daemon serve       # 服务跑在本终端
    python tmp/q25_funnel/run.py --daemon status | stop

    # 云端 GPU(remote.py):接上之后上面所有命令原样用,回测在云端算
    python tmp/q25_funnel/run.py --daemon attach [host]   # host = mcp-ssh/hosts.json 的名字,默认 quant-server
    python tmp/q25_funnel/run.py --daemon detach          # 断开,云端服务留着(stop = 连云端服务一起停)

首次调用可自动启动服务。实时面板 = `python -m monitoring`,数据源是服务写的 `.q25d_status.json`
(云端的由 scripts/mirror_q25_status.py 推到本地);面板按提交文件所在目录把各轮归到提交者名下。

回测由**常驻服务**执行(daemon.py):面板与 CUDA context 只装载一次,所有客户端的请求在
服务端排成单队列逐个算。CUDA 是唯一计算后端；本机直算也要求 CUDA。

Sharpe 使用 qnt Q25 算术年化口径（365 天）。研究窗口来自 config.ini [factor_mining]。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import client


def _fields():
    import engine
    p = Path(__file__).parent / 'datasets.json'
    cats = json.loads(p.read_text(encoding='utf-8'))['categories'] if p.exists() else {}
    print(f'词表 {len(engine.operands())} 个 operand(OHLCV 不入词表,仅供回测定价)\n')
    for c, m in cats.items():
        lvl = m['level_measured']
        print(f'### {c}  [{lvl}]  {m["desc"]}')
        if lvl == 'market':
            print('    ⚠ 市场级:跨资产恒等 ⇒ 只能表达**择时**,不能表达选币')
        for f, v in m['fields'].items():
            rng = '天然∈[0,1] 可裸用作敞口' if v['in_unit_range'] else \
                  f'p1={v["p1"]:.4g} p50={v["p50"]:.4g} p99={v["p99"]:.4g}'
            warn = '  ⚠宽域水平量,请检查 clip 后是否恒满仓' \
                if (not v['in_unit_range'] and abs(v['p99']) > 100) else ''
            print(f'    {f:14s} cov{v["coverage"]:.3f} warmup{v["warmup_row"]:<4d} {rng}{warn}')
        print()


def _ops():
    from fastexpr import MAX_COST,MIN_WINDOW,MAX_WINDOW,ALL_OPS,ARITY
    for arity in sorted(set(ARITY.values())):
        print(f'{arity} 参数: '+ ' '.join(sorted(k for k in ALL_OPS if ARITY[k]==arity)))
    print(f'\n整数窗口 {MIN_WINDOW}..{MAX_WINDOW}；ts_rank 因果门要求窗口 ≤31')
    print('常数参数连续取值，接受语法和数值有效性检查。')
    print(f'cost 上限   {MAX_COST}')
    print('\n敞口语义:表达式输出 → clip[0,1] = 每资产敞口 → W = 敞口/当日流动名数(≈9.6)。'
          '\n            NaN / 非流动自动置 0(留现金)。**符号是一等参数**:负值全被 clip 掉,'
          '\n            方向写反 = 空仓。')


def _report(row, verbose=True):
    print(f'{"✅" if row["passed"] else "❌"} {row["expr"]}')
    if not row['ok']:
        print(f'    {row["error"]}')
        return
    print(f'    sharpe {row["sharpe"]:+.4f}  CAGR {row["cagr"]:+.1%}  turnover {row["turnover"]:.4f}  '
          f'exposure {row["exposure"]:.3f}  maxDD {row["maxdd"]:+.3f}  '
          f'margin {row["margin_bps"]:.0f}bps  年化 {row["ann_ret"]:+.1%}')
    print(f'    {row["explain"]}')
    if (c := row.get('combo')):
        print(f'    当前组合 K={c["n"]} · 候选与组合 ρ={c["rho"]:.3f}(仅诊断)')
        if 'sharpe_after' in c:
            label = '可行权重' if c['feasible'] else '约束违反最小的试探点(未通过)'
            print(f'    {label} {c["w"]:.3f}:近期 Sharpe {c["sharpe"]:.3f} → {c["sharpe_after"]:.3f}; '
                  f'全期 Sharpe {c["full_sharpe"]:.3f} → {c["full_sharpe_after"]:.3f}; '
                  f'CAGR {c["cagr"]:.1%} → {c["cagr_after"]:.1%}; '
                  f'回撤 {c["maxdd"]:.1%} → {c["maxdd_after"]:.1%}')
    if verbose:
        if row.get('yearly'):
            print('    逐年 Sharpe ' + ' '.join(f'{y["year"]}:{y["sharpe"]:+.2f}' for y in row['yearly']))
    if row.get('accepted'):
        print(f'    → 已按权重 {row["weight"]:.3f} 并入组合库')


def _local_eval(exprs, accept=False, window='is'):
    """本机 CUDA 入口，与常驻服务共用批量回测和组合准入。"""
    import engine
    import gates
    from combolib import ComboLib
    from fastexpr import validate
    combo = ComboLib.load()
    ref = combo.ref(window)
    vocab = engine.operands()
    rows = {}
    if not accept:
        uniq = list(dict.fromkeys(exprs))
        for lo in range(0,len(uniq),engine.CUDA.candidate_batch):
            batch = engine.simulate_many(uniq[lo:lo+engine.CUDA.candidate_batch],window)
            for r,g in zip(batch,gates.run_batch(batch,ref)):
                rows[r['expr']] = gates.rowify(r,g)
        return [rows[s] for s in exprs]
    for s in dict.fromkeys(exprs):
        v = validate(s, vocab)
        if not v['ok']:
            rows[s] = gates.reject_row(s, v['error'])
            continue
        r = engine.simulate(s, window)
        g = gates.run(r, ref)
        rows[s] = gates.rowify(r, g)
        if accept and g['passed']:
            rows[s].update(accepted=True, weight=combo.add(r, g))
            ref = combo.ref(window)
    if accept and any(r['accepted'] for r in rows.values()):
        combo.save()
    return [rows[s] for s in exprs]


def _seeds(path):
    """Expression file -> [{expr, hypothesis}]: a `#` line is the hypothesis of every formula below it."""
    seeds, hypothesis = [], ''
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.startswith('#'):
            hypothesis = line.lstrip('#').strip()
        elif line:
            seeds.append(dict(expr=line, hypothesis=hypothesis))
    return seeds


def _family_report(res):
    num = lambda x, spec='+.3f': '—' if x is None else format(x, spec)
    delta = lambda r: f'组合 ΔSharpe {num(r["gain"])}  ΔCAGR {num(r["d_cagr"], "+.1%")}  Δ回撤 {num(r["d_maxdd"], "+.1%")}'
    spread = lambda r: (f'ΔSharpe 中位 {num(r["median"])}(p25 {num(r["p25"])}, p75 {num(r["p75"])})  '
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
    verdicts = [f['verdict'] for f in res['families']]
    print(f'\n汇总:{len(verdicts)} 条假设公式,成立 {verdicts.count("成立")},稀疏 {verdicts.count("稀疏")},尖峰 {verdicts.count("尖峰")},'
          f'无效 {verdicts.count("无效")};本轮共评价 {res["variants"]:,} 个变体(多重检验计数)')


def _local_python(path, window='is', accept=False):
    from combolib import ComboLib, evaluate_python_source
    return evaluate_python_source(Path(path).read_text(encoding='utf-8'),str(path),window,ComboLib.load(),accept)


def _daemon_cmd(cmd, host=None):
    import daemon as D
    if cmd == 'attach':                   # 接到云端服务:先停本地那份(显存让出来),再写端口文件
        import remote
        d = client.info()
        if d and not d.get('host'):
            print(client.request(dict(op='stop'), timeout=10.0) or '(无本地服务)')
        remote.attach(host or remote.DEFAULT_HOST)
        return
    if cmd == 'detach':                   # 只忘掉云端,不停它 —— 别的工作区可能还接着
        client.PORTFILE.unlink(missing_ok=True)
        print('已断开;云端服务未动(要停用 --daemon stop)')
        return
    if cmd == 'serve':                    # 服务跑在**本终端**:先停掉既有的,避免两份显存
        d = client.info()
        if not (d and d.get('host')):     # 接着云端时只断开:云端那份不占本机显存,不该被顺手杀掉
            print(client.request(dict(op='stop'), timeout=10.0) or '(无既有服务)')
        client.PORTFILE.unlink(missing_ok=True)
        D.Server(0).run()
        return
    if cmd == 'stop':
        print(client.request(dict(op='stop'), timeout=10.0) or '未运行')
        client.PORTFILE.unlink(missing_ok=True)
        return
    p = client.request(dict(op='ping'), timeout=10.0)      # status
    print(json.dumps(p, ensure_ascii=False, indent=2) if p else '未运行')


def main():
    ap = argparse.ArgumentParser(prog='run.py', description='Q25 alpha 运行器')
    ap.add_argument('exprs', nargs='*', help='FastExpr 表达式(可多条)')
    ap.add_argument('--file', help='表达式文件,一行一条,# 开头为注释')
    ap.add_argument('--python', help='Python alpha 文件(定义 alpha(data, liquid))')
    ap.add_argument('--family', help='假设文件:`# 假设` 注释 + 公式;每条公式展开成邻域判整片(family.py)')
    ap.add_argument('--fields', action='store_true', help='列出词表与字段实测统计')
    ap.add_argument('--ops', action='store_true', help='列出算子/窗口/常数池/敞口语义')
    ap.add_argument('--json', help='机读结果输出路径')
    ap.add_argument('--accept', action='store_true', help='把全过者按最优权并入组合库(主控专用)')
    ap.add_argument('--quiet', action='store_true', help='结果只打一行摘要')
    ap.add_argument('--no-daemon', action='store_true', help='强制进程内直算,不用常驻服务')
    ap.add_argument('--daemon', choices=['serve', 'status', 'stop', 'attach', 'detach'],
                    help='serve=服务跑在本终端;attach [host]=接到云端 GPU 服务;detach=断开')
    a = ap.parse_args()

    if a.daemon:
        _daemon_cmd(a.daemon, a.exprs[0] if a.exprs else None)
        return
    if a.no_daemon and (client.info() or {}).get('host'):
        sys.exit(f'已接云端 {client.info()["host"]}:--no-daemon 会创建本机 CUDA 上下文,拒绝。'
                 f'回测一律走云端;要本机算先由主控 `--daemon detach`')
    if a.fields:
        _fields()
        return
    if a.ops:
        _ops()
        return

    if a.family:
        seeds = _seeds(a.family)
        if not seeds:
            ap.error(f'{a.family} 里没有公式')
        res = None if a.no_daemon else client.evaluate_family(seeds, Path(a.family).as_posix(), quiet=a.quiet)
        if res is None:
            import family
            from combolib import ComboLib
            res = family.expand(seeds, ComboLib.load().ref('is'))
        _family_report(res)
        if a.json:
            Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        return

    if a.python:
        rows = None if a.no_daemon else client.evaluate_python(Path(a.python), quiet=a.quiet, accept=a.accept)
        if rows is None:
            rows = _local_python(a.python, accept=a.accept)
    else:
        exprs = list(a.exprs)
        if a.file:
            exprs += [l.strip() for l in Path(a.file).read_text(encoding='utf-8').splitlines()
                      if l.strip() and not l.strip().startswith('#')]
        if not exprs:
            ap.error('没有输入:给表达式、--file 或 --python')
        rows = None if a.no_daemon else client.evaluate(
            exprs, accept=a.accept, quiet=a.quiet, source=a.file and Path(a.file).as_posix())
        if rows is None:
            rows = _local_eval(exprs, accept=a.accept)

    # `--quiet` = **只打过门的**,其余按层计数。它以前只是省掉逐年那一行,每条照打三行:
    # 2,132 条候选打出 499KB,面板被淹在里面。海选轮要看的是"过了几条、其余死在哪层",
    # 逐条死因在 `--json` 里,不该刷屏。
    for row in rows:
        if row['passed'] or not a.quiet:
            _report(row, not a.quiet)
    n_pass = sum(bool(r['passed']) for r in rows)
    if a.quiet:
        import collections
        t = collections.Counter(r['stopped_at'] or 'PASS' for r in rows)
        names = {'G1': '结构/因果', 'G2': '组合 Sharpe / 收益', 'PASS': '全过'}
        print('  死因分布  ' + '   '.join(f'{names.get(k, k)} {t[k]:,}'
                                          for k in ('G1', 'G2', 'PASS') if t[k]))
    print(f'\n汇总:{len(rows)} 条,{n_pass} 条通过组合准入')
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

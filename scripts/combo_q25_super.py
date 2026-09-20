"""Q25 组合优化 v2:候选流直接更新组合,没有独立 alpha 准入池。

用法:python scripts/combo_q25_super.py [候选文件.json或.txt]
JSON = [{expr, lane}];Python 的 expr 是 q25_funnel 下相对路径:函数名。
缺省读取 r6/full_pool.json 全部源候选,不读取旧门的通过标签。按输入顺序逐条试探,
每次都调用与 run.py / daemon 相同的 gates.run 和 ComboLib.add。
空组合从等权基准开始,权重非负且和为 1。达到成员上限即停止并明确报告未评估数。
顺序会影响贪心结果;不宣称全局最优。历史候选生成看过全期,前推不是独立样本外证明。
输出 tmp/q25_combo_v2,不会覆盖已有组合库、旧报告或提交 notebook。
现有库的增量优化使用 run.py --accept,保持同一准入函数。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tmp/q25_funnel'))

import numpy as np
import cupy as cp
import pandas as pd
import engine
import gates
from backtest.ops_cuda.q25 import cummax
from combolib import ComboLib, SEED, member_book
from engine import PPY, _metrics, official_sharpe

FUNNEL = REPO / 'tmp/q25_funnel'
OUT = REPO / 'tmp/q25_combo_v2'
FOLDS = (('C', '2020-09-01', '2022-09-01'), ('A', '2022-09-01', '2024-09-01'),
         ('B', '2024-09-01', '2026-09-01'))


def test_row(rr: np.ndarray, d: pd.DatetimeIndex, lo: str, hi: str):
    """测试窗 (lo, hi] → (两年 Sharpe, 第 1 年, 第 2 年, 两年均值, CAGR, 最大回撤)。"""
    def win(a, b):
        return np.asarray((d > pd.Timestamp(a)) & (d <= pd.Timestamp(b)))
    mid = str((pd.Timestamp(lo) + pd.DateOffset(years=1)).date())
    a, b = official_sharpe(rr[win(lo, mid)]), official_sharpe(rr[win(mid, hi)])
    lg = cp.log1p(rr[win(lo, hi)])
    eq = cp.cumsum(lg)
    dd = float(cp.expm1((eq - cp.maximum(cummax(eq), 0.0)).min()))
    return official_sharpe(rr[win(lo, hi)]), a, b, (a + b) / 2, float(cp.expm1(lg.sum() * PPY / len(lg))), dd


def load_pool(path=None):
    """候选输入是源表达式,绝不按旧 ok/GR/G3 标签过滤。"""
    if path is not None:
        path = Path(path)
        if path.suffix == '.json':
            rows = json.loads(path.read_text(encoding='utf-8'))
            return [dict(expr=r['expr'], lane=r.get('lane', 'fastexpr')) for r in rows]
        return [dict(expr=s.strip(), lane='fastexpr') for s in path.read_text(encoding='utf-8').splitlines()
                if s.strip() and not s.lstrip().startswith('#')]
    return load_pool(FUNNEL/'candidates/r6/full_pool.json')



def prepare(cut: str) -> str:
    """截断数据,选权重时看不到测试段;每个窗口有独立缓存键。"""
    data = engine.load()
    full = data['windows']['is']
    end = int(np.searchsorted(full['dates'], np.datetime64(cut), side='right'))
    assert end > full['row0']
    key = f'combo-v2-{cut}'
    data['windows'][key] = dict(
        panels={k: cp.ascontiguousarray(v[:end]) for k, v in full['panels'].items()},
        static={k: cp.ascontiguousarray(v[:end]) for k, v in full['static'].items()},
        liquid=full['liquid'][:end], dates=full['dates'][:end], row0=full['row0'],
        span=(full['span'][0], cut))
    engine._BM.pop(key, None)
    return key


def evaluate_candidate(item, window):
    """两种车道均走 engine 的结构/因果验证。"""
    if item['lane'] != 'python':
        why = gates.g1_causality(item['expr'])
        if why:
            return dict(ok=False, error=why, expr=item['expr'], lane=item['lane'])
        r = engine.simulate(item['expr'], window)
    else:
        rel, name = item['expr'].rsplit(':', 1)
        path = FUNNEL / rel
        from alpha_cuda import load_source
        ns = load_source(path.read_text(encoding='utf-8'), str(path))
        r = engine.simulate_python(ns[name], window)
        r['expr'] = item['expr']
    r['lane'] = item['lane']
    return r


def build(candidates, window='is', combo=None, progress=None):
    """按候选流更新当前组合;返回精算前后指标和每一步的拒绝/接受原因。"""
    combo = ComboLib([]) if combo is None else combo
    trace = []
    seen = set()
    for i, item in enumerate(candidates):
        if len(combo.members) >= gates.CFG.max_members:
            break
        identity = (item['expr'], item['lane'])
        if identity in seen:
            continue
        seen.add(identity)
        r = evaluate_candidate(item, window)
        verdict = gates.run(r, combo.ref(window))
        row = gates.rowify(r, verdict)
        if verdict['passed']:
            row.update(accepted=True, weight=combo.add(r, verdict))
        trace.append(row)
        if progress is not None:
            progress(i + 1, len(candidates), row)
    return combo, trace


def fold(candidates, cut):
    return build(candidates, prepare(cut))


def main():
    t0 = time.time()
    candidates = load_pool(sys.argv[1] if len(sys.argv) > 1 else None)
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    def progress(i, total, row):
        if row['accepted'] or i % 100 == 0:
            print(f'{i}/{total} · {row["explain"]} · {time.time()-t0:.0f}s', flush=True)
    lines = ['# Q25 组合优化 v2', '', '所有接受步骤都按实际合成持仓扣成本精算;计算全程使用 CUDA FP64。',
             '候选顺序影响结果;历史候选生成使用过全期,以下前推不能当作独立样本外。', '',
             '| 窗口 | 已评估 | 接受 | 成员 | 全期 Sharpe | 近期加权 Sharpe | CAGR | 最大回撤 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    full = engine.load()['windows']['is']
    end = str(full['dates'][-1])
    for tag, cut, test_end in FOLDS + (('full', end, None),):
        window = prepare(cut)
        combo, trace = build(candidates, window, progress=progress)
        m = combo.ref(window)
        members = combo.members or [dict(SEED)]
        record = dict(protocol=gates.PROTOCOL, candidate_count=len(candidates), evaluated=len(trace),
                      remaining=len(candidates)-len(trace), members=members, trace=trace,
                      metrics={k: m[k] for k in ('sharpe', 'wsharpe', 'cagr', 'maxdd')})
        lines.append(f'| {tag} ..{cut} | {len(trace)} | {sum(x["accepted"] for x in trace)} | '
                     f'{len(members)} | {m["sharpe"]:.3f} | {m["wsharpe"]:.3f} | {m["cagr"]:.1%} | {m["maxdd"]:.1%} |')
        if test_end is not None:
            test_window = prepare(test_end)
            test_ref = ComboLib(members).ref(test_window)
            dates = pd.DatetimeIndex(test_ref['win']['dates'][test_ref['win']['row0']:])
            record['forward'] = test_row(test_ref['rr'], dates, cut, test_end)
        else:
            record['submission_ready'] = m['sharpe'] >= gates.CFG.submission_sharpe
            cp.save(OUT / 'W.npy', m['W'])
            (OUT / 'members.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        records[tag] = record
    (OUT / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'完成 {time.time()-t0:.0f}s → {OUT}', flush=True)


if __name__ == '__main__':
    main()

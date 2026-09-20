"""组合库 —— 组合本身就是库(2026-09-19 用户:"最后提交的 alpha 肯定是组合 alpha,组合 alpha 和单 alpha
挖掘可以合并";取代 SC 库 sclib.py / sc_library.json)。

库 = 当前组合:成员 + 常数权重(非负、和为 1)。组合书 = Σ α_i·书_i(逐币 ≤ 1/n_liq、总敞口 ≤ 1),
引擎 f64 精算(换手在组合内对冲)。gates.run(r, ref) 搜索并验证实际合成权重;
add(r, verdict) 只接受这个参照上已通过的权重。空库从满仓等权基准开始。

写库的两处:scripts/combo_q25_super.py 定稿时整体写出;常驻服务 / run.py 的 --accept 逐条并入。
常驻服务按文件 mtime 重载。Python 成员的 expr = 'tmp/q25_funnel 下的相对路径:函数名'。
"""
from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))


STORE = HERE / 'combo_library.json'     # 引擎 / 门延迟导入:remote.py 只要这个路径,不该为它装引擎
SEED = dict(expr='__equal_weight__', lane='benchmark', weight=1.0)


def member_book(m: Dict, window: str):
    import engine
    win=engine.load()['windows'][window]
    if m['lane']=='benchmark':
        from combo_gpu import device_window
        g=device_window(win)
        return g.dliq/g.nliq
    if m['lane']=='python':
        from alpha_cuda import load_source
        rel,name=m['expr'].rsplit(':',1)
        path=HERE/rel
        ns=load_source(path.read_text(encoding='utf-8'),str(path))
        return engine.simulate_python(ns[name],window,check_lookahead=False)['_W']
    return engine.simulate(m['expr'],window)['_W']


def evaluate_python_source(source: str, path: str, window: str, combo, accept=False, names=None):
    """Python 与表达式走同一 gate;接受时保存内容寻址的自包含源代码以便重建持仓。"""
    import engine
    import gates
    from alpha_cuda import load_source
    ns=load_source(source,path)
    fns = [(k,v) for k,v in ns.items() if k.startswith('alpha') and callable(v) and (names is None or k in names)]
    assert fns, 'Python 源必须定义 alpha 开头的函数'
    rel = f'candidates/accepted_python/{hashlib.sha256(source.encode()).hexdigest()}.py'
    rows = []
    for name,fn in fns:
        r = engine.simulate_python(fn,window)
        r['expr'] = f'{rel}:{name}'
        g = gates.run(r,combo.ref(window))
        row = gates.rowify(r,g)
        if accept and g['passed']:
            target = HERE / rel
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(source,encoding='utf-8')
            row.update(accepted=True,weight=combo.add(r,g))
        rows.append(row)
    if any(r['accepted'] for r in rows):
        combo.save()
    return rows


class ComboLib:
    def __init__(self, members: List[Dict]):
        self.members = members                    # [dict(expr, lane, weight)]
        self._ref: Dict[str, Dict] = {}

    @classmethod
    def load(cls) -> 'ComboLib':
        if not STORE.exists():
            return cls([])
        return cls(json.loads(STORE.read_text(encoding='utf-8'))['members'])

    def save(self) -> None:
        STORE.write_text(json.dumps(dict(updated=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                         members=self.members), ensure_ascii=False, indent=1), encoding='utf-8')

    def ref(self, window: str = 'is') -> Optional[Dict]:
        """精算参照包含实际持仓 W;空库返回等权种子,不写磁盘。"""
        if window not in self._ref:
            import engine
            import gates
            from engine import _metrics
            win = engine.load()['windows'][window]
            members = self.members or [SEED]
            import cupy as cp
            W=cp.sum(cp.stack([member_book(x,window) for x in members])*
                     cp.asarray([x['weight'] for x in members])[:,None,None],axis=0)
            self._ref[window] = gates.make_ref(W, win, len(members), seed=not self.members)
            self._ref[window]['member_keys'] = {(m['expr'],m['lane']) for m in members}
        return self._ref[window]

    def add(self, r: Dict, verdict: Dict) -> float:
        """只能使用当前参照上已经精算通过的权重;拒绝过期判定。"""
        ref = self.ref(r['window'])
        proposal = verdict['layers']['G2_performance']
        assert proposal['backend'] == 'cuda-f64', 'Unrefined precision cannot update the portfolio'
        assert verdict['passed'] and proposal['_ref'] is ref and verdict['expr'] == r['expr']
        w = proposal['weight']
        assert 0 < w <= 1
        members = self.members or [SEED]
        self.members = [dict(m, weight=m['weight'] * (1.0 - w)) for m in members if w < 1.0]
        lane = r.get('lane', 'fastexpr')
        for m in self.members:
            if m['expr'] == r['expr'] and m['lane'] == lane:
                m['weight'] += w
                break
        else:
            self.members.append(dict(expr=r['expr'], lane=lane, weight=w))
        self._ref.clear()
        return w

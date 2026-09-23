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
import os
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))


STORE = HERE / 'combo_library.json'     # 引擎 / 门延迟导入:remote.py 只要这个路径,不该为它装引擎
SEED = dict(expr='__equal_weight__', lane='benchmark', weight=1.0)


_STATES: Dict = {}     # (expr, lane, window) -> (持仓书, 净日收益):常驻显存,成员一经算出不再重算


def member_states(members: List[Dict], window: str) -> List:
    """成员的 (持仓书, 净日收益),常驻显存。未缓存的表达式成员合成一批一次算完。
    2026-09-21 实测(V100S,空库起步 60 轮):此前每轮为每个成员单独重跑"解析→求值→回测→报表",
    占整轮墙钟 67%(`Evaluator()` 0.66 s/次),成员越多越慢;成员的书只取决于表达式与数据,无须重算。"""
    import engine
    win = engine.load()['windows'][window]                                # 同时按显存定下 candidate_batch
    missing = [m for m in members if (m['expr'], m['lane'], window) not in _STATES]
    exprs = [m for m in missing if m['lane'] not in ('benchmark', 'python')]
    for lo in range(0, len(exprs), engine.CUDA.candidate_batch):          # 批上限来自显存,不是成员数
        part = exprs[lo:lo+engine.CUDA.candidate_batch]
        for m, r in zip(part, engine.simulate_many([m['expr'] for m in part], window)):
            _STATES[(m['expr'], m['lane'], window)] = (r['_W'], r['_rr'])
    for m in missing:
        if m['lane'] == 'benchmark':
            from combo_gpu import device_window
            g = device_window(win)
            book = g.dliq/g.nliq
            _STATES[(m['expr'], m['lane'], window)] = (book, engine._metrics(book, win)['rr'])
        elif m['lane'] == 'python':
            from alpha_cuda import load_source
            rel, name = m['expr'].rsplit(':', 1)
            path = HERE/rel
            r = engine.simulate_python(load_source(path.read_text(encoding='utf-8'), str(path))[name], window, check_lookahead=False)
            _STATES[(m['expr'], m['lane'], window)] = (r['_W'], r['_rr'])
    return [_STATES[(m['expr'], m['lane'], window)] for m in members]


def member_book(m: Dict, window: str):
    return member_states([m], window)[0][0]


def remember(expr: str, lane: str, window: str, book, rr) -> None:
    """准入时书与收益已经算出:直接入缓存,入库后不再重算。"""
    _STATES[(expr, lane, window)] = (book, rr)


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
        self._disk_snapshot = None

    @classmethod
    def load(cls) -> 'ComboLib':
        content = STORE.read_bytes() if STORE.exists() else b''
        lib = cls(json.loads(content)['members'] if content else [])
        lib._disk_snapshot = content
        return lib

    def save(self) -> None:
        lock = STORE.with_suffix('.lock')
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        temp = STORE.with_suffix(f'.{os.getpid()}.tmp')
        try:
            os.close(fd)
            current = STORE.read_bytes() if STORE.exists() else b''
            if self._disk_snapshot is not None and current != self._disk_snapshot:
                raise RuntimeError('Portfolio changed on disk; refusing stale search update')
            content = json.dumps(dict(updated=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                      members=self.members), ensure_ascii=False, indent=1).encode('utf-8')
            temp.write_bytes(content)
            os.replace(temp, STORE)
            self._disk_snapshot = content
        finally:
            temp.unlink(missing_ok=True)
            lock.unlink()

    def ref(self, window: str = 'is') -> Optional[Dict]:
        """精算参照包含实际持仓 W;空库返回等权种子,不写磁盘。"""
        if window not in self._ref:
            import engine
            import gates
            from engine import _metrics
            win = engine.load()['windows'][window]
            members = self.members or [SEED]
            import cupy as cp
            U=cp.sum(cp.stack([book for book,_ in member_states(members,window)])*
                     cp.asarray([x['weight'] for x in members])[:,None,None],axis=0)
            self._ref[window] = gates.make_ref(U, win, len(members), seed=not self.members)
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
        # 权重是各成员书的相加倍数(不是份额):新成员直接加入,已有成员原样不动。
        self.members = [dict(m) for m in self.members or [SEED]]
        lane = r.get('lane', 'fastexpr')
        remember(r['expr'], lane, r['window'], r['_W'], r['_rr'])
        for m in self.members:
            if m['expr'] == r['expr'] and m['lane'] == lane:
                m['weight'] += w
                break
        else:
            self.members.append(dict(expr=r['expr'], lane=lane, weight=w))
        self._ref.clear()
        return w

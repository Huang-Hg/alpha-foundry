"""项目入口 — Q25 crypto_daily 时序书因子挖掘。

每个子命令只接位置参数(输出路径);所有时间区间 / 超参 / 数据路径都在 `config/config.ini`。
改 INI 即改全局,不再有 CLI 覆盖分支。

用法:
    python main.py gp-baseline   OUTPUT_JSON      # DEAP 强类型 GP 因子挖掘(NSGA-II)+ alpha pool
    python main.py alphasage     OUTPUT_JSON      # AlphaSAGE GFlowNet 因子挖掘 + alpha pool

gp-baseline / alphasage = 搜索 + edge 准入建池(成员 gate 等权 ensemble);评估全复用
evaluator/alpha_pool。旧入口 2026-07-22 退役:gp-equity / gp-ashare(时序书口径下 evaluator
直接 raise,死入口;重启从 git 历史恢复)、materialize(永续 5m 物化;其全栈 2026-07-25 删除)。
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# kill -SIGUSR1 PID → 把全 Python thread 栈打到 stderr(诊断 deadlock/hang;不停进程)
# faulthandler.register + SIGUSR1 仅 Unix 有(OS 系统边界差异);Windows 本地无 → 跳过(server Linux 照常注册)。
if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

# libgomp 空闲线程默认 spin-wait(ACTIVE):self-play worker 的 OMP 线程在主进程**串行训练段**仍
#   空转抢核(实测训练段 worker 占 6-9 核)→ tree-LSTM 训练被饿死 ~40×(2026-06-22 实锤 train
#   290s→7.7s,sp 无损)。passive=空闲即睡。**必须在 torch/numpy/C 扩展(libgomp init)导入前设。**
os.environ.setdefault('OMP_WAIT_POLICY', 'passive')
os.environ.setdefault('GOMP_SPINCOUNT', '0')

import numpy as np

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)sZ %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.info

from config.config import FactorMiningConfig, ini
from evaluation.expression import evaluate as eval_tree
from rl.backtest_reward import BacktestRewardConfig
from rl.evaluator import _DEVICE as _EVAL_DEVICE


# ============================================================================
# gp-baseline — DEAP 强类型 GP 因子挖掘(NSGA-II 双目标;评估全复用共享管线)
#   2026-07-04 起 crypto 挖矿 = Q25 crypto_daily(Quantiacs 日线,回测/reward = qnt 1:1);
#   旧永续 5m 挖矿口径(adapter/引擎/词表全栈)2026-07-25 已彻底删除。
# ============================================================================

# 数据切分(INI [factor_mining]):test = 竞赛 live,天然 held out。
_fm_win = FactorMiningConfig()
_GP_TRAIN = (_fm_win.train_start, _fm_win.train_end)
_GP_VAL   = (_fm_win.val_start, _fm_win.val_end)


def prepare_gp_data(train_start: str, train_end: str, val_start: str, val_end: str,
                    data_dir: str) -> Dict[str, Any]:
    """Q25 crypto_daily(Quantiacs)挖矿数据:data_dir 下三源同网格 ——
    panel.parquet(scripts/build_q25_dataset.py:OHLCV+is_liquid,列 {field}__{asset})
    + features.parquet(同脚本:OHLCV 提取分解层 = scale-free 原语
    resret/rangepos/gkvol… + 因果滚动 PLS 正交复合 plsa..plsd;2026-07-05 池审计 N_eff=1.2
    修复,同日 h* 截面字段证否删除 —— 流动宇宙覆盖仅 ~40% 致搜索 0/24 使用)
    + market.parquet(市场级序列:BTC 链上/股指/BTC regime/alt_rv,广播成 (T,S) operand
    面板)→ 网格 = 数据头..train_end(ts 因子 warmup 完整且因果)+ evaluator qnt 奖励上下文
    (set_qnt_context)+ val 件(网格 = 数据头..val_end,回测只看 val 窗)。词表 = 特征列 +
    事件列 + 趋势列 + 市场列(列序 = 导出序,产生式 ID 稳定),窗集切 crypto_daily。
    前向标签管线(train_y/val_y,截面 IC 口径遗留)2026-07-22 随冗余机制清除删除。"""
    from markets.profile import get_profile
    from markets.vocabulary import from_columns
    from evaluation.grammar import set_vocabulary, set_windows
    from rl.backtest_reward import prepare_qnt_static
    from rl.evaluator import set_qnt_context
    import pyarrow.parquet as pq
    ptbl = pq.read_table(Path(data_dir) / 'panel.parquet')
    mtbl = pq.read_table(Path(data_dir) / 'market.parquet')
    ftbl = pq.read_table(Path(data_dir) / 'features.parquet')
    assert ptbl.column('date').to_pylist() == mtbl.column('date').to_pylist()  # 文件边界:三源同网格
    assert ptbl.column('date').to_pylist() == ftbl.column('date').to_pylist()
    dates = np.array(ptbl.column('date').to_pylist(), dtype='datetime64[D]')
    assets = [n.split('__', 1)[1] for n in ptbl.column_names if n.startswith('close__')]

    def grid(field: str, tbl=ptbl) -> np.ndarray:
        return np.ascontiguousarray(
            np.column_stack([tbl.column(f'{field}__{a}').to_numpy() for a in assets]))

    o, h, l, c, v = (grid(k) for k in ('open', 'high', 'low', 'close', 'vol'))
    liquid = grid('liquid') > 0.0
    extra: List[str] = []
    for n in ftbl.column_names:                               # 特征字段序 = 派生脚本序(产生式 ID 稳定)
        f_ = n.split('__', 1)[0]
        if '__' in n and f_ not in extra:
            extra.append(f_)
    # 事件面板(可选,文件 IO 系统边界):scripts/build_unlock_events.py 从公开 vesting 日程产
    # events.parquet(d2unlock/unlockpct/unlockwin 按币种字段,决策时可知无 lookahead)→ 入词表,
    # 公式可与趋势状态自由复合(时序书口径的事件融合第一路;第二路=部署层 unlock_downweight)。
    evt: List[str] = []
    etbl = None
    epath = Path(data_dir) / 'events.parquet'
    if epath.exists():
        etbl = pq.read_table(epath)
        assert etbl.column('date').to_pylist() == ptbl.column('date').to_pylist()   # 同网格
        for n in etbl.column_names:
            f_ = n.split('__', 1)[0]
            if '__' in n and f_ not in evt:
                evt.append(f_)
    market = {n: mtbl.column(n).to_numpy() for n in mtbl.column_names if n != 'date'}
    T, S = c.shape
    r_tr0 = int(np.searchsorted(dates, np.datetime64(train_start)))
    r_tr1 = int(np.searchsorted(dates, np.datetime64(train_end), side='right'))
    r_va0 = int(np.searchsorted(dates, np.datetime64(val_start)))
    r_va1 = int(np.searchsorted(dates, np.datetime64(val_end), side='right'))
    # 趋势/动量 operand(2026-07-06:补 miner 缺的动量原料 —— 词表原本只 OHLCV+特征+市场、无任何趋势轴,
    #   动量须从 close 现搭 ≥3-production 树,GFN 采短树(均长 3.3)+ max_len=5 够不到 → 系统性挖不到动量。
    #   补 ma_dist_w=close/SMA_w−1(动量本体)+ ret_w=w 日收益后,momentum=ma_dist_20+ma_dist_120 可 1-3
    #   production 表达,GFN 轻松采到、超额 IR reward 已会奖励。因果:SMA/shift 只用 close[≤t])。
    import pandas as pd
    _cdf = pd.DataFrame(c)
    trend: Dict[str, np.ndarray] = {}
    for _w in (20, 60, 120):
        _sma = _cdf.rolling(_w, min_periods=_w).mean().to_numpy()
        _prev = _cdf.shift(_w).to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            trend[f'ma_dist_{_w}'] = np.ascontiguousarray(np.where(_sma > 0, c / _sma - 1.0, np.nan))
            trend[f'ret_{_w}'] = np.ascontiguousarray(np.where(_prev > 0, c / _prev - 1.0, np.nan))
    prof = get_profile('crypto_daily')
    # 2026-07-06(B):删原始 OHLCV operand —— 跨币 12-14 数量级(逼 f64、cs_rank 无意义=名义价大小、G0 已挡纯
    #   wrapper),有用信息已全被 scale-free 特征覆盖(ret/ma_dist→close 动态,rangepos/rangerel/candlebody/
    #   gkvol→OHLC 日内,dvol/amihud→量)。OHLCV 仅作 feature/trend 派生 + 回测引擎价格
    #   (留 panels_full),不再入 DSL 词表 → operand 全 scale-free、因子求值可 f32。
    set_vocabulary(from_columns([*extra, *evt, *trend, *market], name=prof.name, require_base=False))
    set_windows(prof.windows)                                     # 日线窗集(3..240 交易日)

    def _feat(k):                                                 # amihud=illiquidity 比率(dvol 分母→跨币 ~10 数量级),
        arr = grid(k, ftbl)                                       # log 压成 O(1) f32-safe(2026-07-06 B:唯一残余宽域 operand)
        if k == 'amihud':
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.ascontiguousarray(np.where(arr > 0, np.log(arr), np.nan))
        return arr
    panels_full = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
                   **{k: _feat(k) for k in extra},
                   **{k: grid(k, etbl) for k in evt}, **trend,
                   **{k: np.ascontiguousarray(np.broadcast_to(mv[:, None], (T, S)))
                      for k, mv in market.items()}}
    train_panels = {k: np.ascontiguousarray(p[:r_tr1]) for k, p in panels_full.items()}
    # evaluator qnt 奖励上下文:静态量网格 = train 网格(reward 统计窗 [r_tr0:r_tr1),无 val 泄漏)
    set_qnt_context(prepare_qnt_static(o[:r_tr1], h[:r_tr1], l[:r_tr1], c[:r_tr1]),
                    liquid[:r_tr1], r_tr0)
    # val 件:网格 = 数据头..val_end(成员因子 warmup 完整、因果),回测只看 val 窗
    val_panels = {k: np.ascontiguousarray(p[:r_va1]) for k, p in panels_full.items()}
    log(f"[prepare] qnt panel T={T} S={S} market_cols={len(market)} "
        f"train[{r_tr0}:{r_tr1}) val[{r_va0}:{r_va1}) liquid/day={liquid.sum(1).mean():.1f}")
    return dict(train_panels=train_panels, val_panels=val_panels,
                val_row0=r_va0, val_liquid=liquid[:r_va1],
                static_val=prepare_qnt_static(o[:r_va1], h[:r_va1], l[:r_va1], c[:r_va1]),
                assets=assets)


def _val_book(trees: List, data: Dict[str, Any], bcfg) -> Dict[str, float]:
    """成员树列表 → val 窗成员 gate 等权书 qnt 1:1 读数(两级等权;eval_pool_val / 二级提交书共用)。
    返回 dict(qnt 官方口径领衔):sharpe / maxdd / ret;ew_sharpe = 流动名 EW 锚同窗(独立参照);
    edge = sharpe − ew_sharpe(挖矿内部 reward 同源,仅作参考)。"""
    from config.config import TsBookConfig
    from rl.backtest_reward import run_bt
    from rl.sizing import qnt_free_exposure, qnt_gate_book
    r0 = data['val_row0']
    liquid = data['val_liquid']
    exps = [qnt_free_exposure(np.ascontiguousarray(
                eval_tree(t, data['val_panels'], None), dtype=np.float64), liquid)
            for t in trees]
    W = qnt_gate_book(np.mean(np.stack(exps), axis=0), liquid, TsBookConfig().gross_scale)
    W[:r0] = 0.0
    res = run_bt(np.ascontiguousarray(W), data['static_val'], r0, bcfg)
    W_ew = np.zeros(liquid.shape, dtype=np.float64)
    L = liquid[r0:].astype(np.float64)
    W_ew[r0:] = L / np.maximum(L.sum(1, keepdims=True), 1.0)
    ew = run_bt(W_ew, data['static_val'], r0, bcfg)
    return dict(sharpe=float(res['sharpe']), maxdd=float(res['max_drawdown']),
                ret=float(res['total_return']), ew_sharpe=float(ew['sharpe']),
                edge=float(res['sharpe'] - ew['sharpe']))


def eval_pool_val(pool, data: Dict[str, Any], bcfg) -> Dict[str, float]:
    """一级池在 val 段的读数(时序书口径,裸 alpha 无 overlay;2026-07-22 截面→时序转向,
    旧 AFF lstsq + softmax 截面书读数随口径退役)。空池 → 全 0。"""
    if not pool.members:
        return dict(sharpe=0.0, maxdd=0.0, ret=0.0, ew_sharpe=0.0, edge=0.0)
    return _val_book([m.tree for m in pool.members], data, bcfg)


# ============================================================================
# 二级提交池信号门(2026-07-23 预注册,用户拍板)
# ============================================================================

_SCISSORS_MAX = 0.5    # 剪刀差上限 |train−val Sharpe|(协议常量,预注册勿炼参,故不入 INI)


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _ledger_append(row: Dict[str, Any]) -> None:
    """val 触碰记账(logs/val_ledger.jsonl,只追加):val = 有限资源,试验数是后续
    DSR 折减 / 审计的分母,凡有表达式或书触碰 val 就落一行。"""
    p = Path('logs/val_ledger.jsonl')
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def submission_gate(members: List, data: Dict[str, Any], bcfg,
                    engine: str, run_ts: str) -> List[Dict[str, Any]]:
    """二级信号门:run 后对一级池成员(≤capacity 个,非 collected 全体 —— 控 val 多重检验)
    逐个读 val 窗,过门 ⇔ ① train edge>0(一级准入已保证)② val Sharpe > val EW 锚
    ③ 剪刀差 |train−val| < _SCISSORS_MAX(双窗复制门:纯噪声双高概率≈单窗平方;
    se(diff)≈0.68 → 真稳定 alpha 约半数随机误杀,拒绝器要精度不要召回,空手=正常预期)。
    红线:本门绝不进 reward / 一级准入(R_NOV 对一级池算,val 无路径回流搜索器)。
    凡经本门选择的对象,val 读数从此只是"通过了门",非无偏 OOS —— 唯一法官 = 前向账本。
    members = [(tree, train_edge)];每成员 val 触碰记账。返回全体 rows(含 passed 标志)。"""
    from rl.backtest_reward import run_bt
    from rl.evaluator import qnt_bench_sharpe
    from rl.sizing import qnt_free_exposure, qnt_gate_book
    r0 = data['val_row0']
    liquid = data['val_liquid']
    W_ew = np.zeros(liquid.shape, dtype=np.float64)
    L = liquid[r0:].astype(np.float64)
    W_ew[r0:] = L / np.maximum(L.sum(1, keepdims=True), 1.0)
    ew_val = float(run_bt(W_ew, data['static_val'], r0, bcfg)['sharpe'])
    bench_tr = qnt_bench_sharpe()
    rows: List[Dict[str, Any]] = []
    now = _now_iso()
    for tree, edge in members:
        E = qnt_free_exposure(np.ascontiguousarray(
            eval_tree(tree, data['val_panels'], None), dtype=np.float64), liquid)
        W = qnt_gate_book(E, liquid)
        W[:r0] = 0.0
        res = run_bt(np.ascontiguousarray(W), data['static_val'], r0, bcfg)
        tr_sh, va_sh = float(edge) + bench_tr, float(res['sharpe'])
        ok = float(edge) > 0.0 and va_sh > ew_val and abs(tr_sh - va_sh) < _SCISSORS_MAX
        rows.append(dict(tree=tree.to_dict(), pretty=tree.pretty(),
                         train_sharpe=tr_sh, val_sharpe=va_sh,
                         val_maxdd=float(res['max_drawdown']), passed=ok))
        _ledger_append({'ts': now, 'engine': engine, 'run': run_ts, 'kind': 'member',
                        'pretty': tree.pretty(), 'train_sharpe': round(tr_sh, 4),
                        'val_sharpe': round(va_sh, 4), 'ew_val': round(ew_val, 4),
                        'passed': ok})
    return rows


def _battery_filter(pool, data: Dict[str, Any], log, tag: str):
    """多矩复合门前置滤器(2026-07-24 安装,回溯标定见 rl.evaluator.moment_battery):
    一级池成员先过 train-only 四矩电池,**全矩通过才触碰 val**(压零假设过门率 + 省触碰)。
    返回 (members=[(tree,q)] 过滤后, battery=全体电池明细供 snapshot 归档)。"""
    from rl.evaluator import moment_battery
    battery = []
    members = []
    for m, q in zip(pool.members, pool.single_q):
        b = moment_battery(m.tree, data['train_panels'])
        battery.append({'pretty': m.tree.pretty(),
                        **{k: (round(float(v), 3) if np.isfinite(v) else None)
                           for k, v in b.items()}})
        if b['n_pass'] == b['n_present']:
            members.append((m.tree, q))
    log(f"[{tag}] 多矩门:{len(members)}/{pool.size} 全矩通过 → 触碰 val"
        f"(拦截 {pool.size - len(members)};规则=全矩,标定 ρ=+0.30)")
    return members, battery


def cmd_gp_baseline(output: str) -> None:
    """DEAP 强类型 GP 因子挖掘:load panel → NSGA-II 搜索(train)→ net 感知准入建池 → val 池回报
    → 落 deploy bundle(output JSON)+ snapshot。device 由 [evaluator] 派发(V100 走 cuda 单进程)。"""
    os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'   # Arrow 还内存(同 train-policy)
    from search.gp.gp_baseline import run_gp
    fm = FactorMiningConfig()
    bcfg = BacktestRewardConfig()
    log(f"[gp-baseline] Q25 crypto_daily train={_GP_TRAIN[0]}~{_GP_TRAIN[1]}  "
        f"val={_GP_VAL[0]}~{_GP_VAL[1]}  (test = 竞赛 live)")
    data = prepare_gp_data(_GP_TRAIN[0], _GP_TRAIN[1], _GP_VAL[0], _GP_VAL[1], fm.qnt_data_dir)
    run_ts = os.environ.get('RUN_TS') or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    snap = Path(f'logs/gp_pool_{run_ts}.jsonl')
    snap.parent.mkdir(parents=True, exist_ok=True)
    seed = int.from_bytes(os.urandom(4), 'little')
    log(f"[gp-baseline] seed={seed}(每次运行随机指定;复现固定此 seed)")
    pool, collected = run_gp(data['train_panels'],
                             capacity=fm.pool_capacity, seed=seed, log=log, snap_path=str(snap))
    v = eval_pool_val(pool, data, bcfg)
    log(f"[gp-baseline] pool_size={pool.size} collected={len(collected)} "
        f"val sharpe={v['sharpe']:+.4f}(EW 锚 {v['ew_sharpe']:+.4f})"
        f"maxDD={v['maxdd']:+.3f} ret={v['ret']:+.4f}")
    members, battery = _battery_filter(pool, data, log, 'gp-baseline')
    sub = submission_gate(members, data, bcfg, 'gp', run_ts)
    passed_trees = [t for (t, _), r in zip(members, sub) if r['passed']]
    if passed_trees:
        sb = _val_book(passed_trees, data, bcfg)
        _ledger_append({'ts': _now_iso(), 'engine': 'gp', 'run': run_ts,
                        'kind': 'submission_book', 'n_members': len(passed_trees),
                        'val_sharpe': round(sb['sharpe'], 4), 'ew_val': round(sb['ew_sharpe'], 4)})
        log(f"[gp-baseline] 二级信号门 {len(passed_trees)}/{len(sub)} 过门  提交书 "
            f"val sharpe={sb['sharpe']:+.4f}(EW 锚 {sb['ew_sharpe']:+.4f})"
            f"maxDD={sb['maxdd']:+.3f}(门后读数,选择偏差已入)")
    else:
        log(f"[gp-baseline] 二级信号门 0/{len(sub)} 过门(双窗复制门空手=正常预期)")
    with open(snap, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'pool': pool.to_jsonable(), 'val': v, 'submission': sub,
                            'battery': battery,
                            'collected': len(collected)}, ensure_ascii=False) + '\n')
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({'pool': pool.to_jsonable(),
                   'submission': [r for r in sub if r['passed']]},
                  f, ensure_ascii=False, indent=2)   # {'pool':[...]} = 部署/materialize 同契约
    log(f"[gp-baseline] deploy bundle → {output};snapshot → {snap}")

    # 跨 run 正交 alpha 库:本轮 collected 内存 pnl 贪心挑 within-run 正交高 IC append(只存因子)
    from rl.alpha_pool import append_run
    lib_path = Path(ini('alpha_library', 'path', 'output/alpha_library.json'))
    n_lib = append_run(collected, lib_path, ini('alpha_library', 'corr_max', 0.5),
                       int(ini('alpha_library', 'max_size', 500)), run_ts)
    log(f"[gp-baseline] alpha library += {n_lib}(within-run 正交)→ {lib_path}")


# ============================================================================
# alphasage — AlphaSAGE GFlowNet alpha 因子挖掘(独立 baseline,与 GP 平级,非改进 GP)
#   数据/切分/val 池回报口径 = 与 gp-baseline 逐字一致 → 平行对比。
#   GFN(torch)走 [evaluator].device(V100 → cuda,本地 → cpu;单一旋钮)。
# ============================================================================

def cmd_alphasage(output: str) -> None:
    """AlphaSAGE GFlowNet baseline:load panel → GFN P(α)∝R 搜索(train,reward=R_IC+nov_weight·R_NOV)
    → LinearAlphaPool(同 GP 的 edge>0 + DPP 正交 Δ 准入)→ val 池回报(eval_pool_val,同 GP judge)
    → 落 deploy bundle(output JSON)+ snapshot。device 由 [evaluator] 派发(GFN+eval 同走)。"""
    os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'   # Arrow 还内存(同 gp-baseline)
    from config.config import AlphaSAGEConfig
    from search.alphasage.gfn_trainer import train_alphasage
    asc = AlphaSAGEConfig()
    fm = FactorMiningConfig()
    bcfg = BacktestRewardConfig()
    log(f"[alphasage] Q25 crypto_daily train={_GP_TRAIN[0]}~{_GP_TRAIN[1]}  "
        f"val={_GP_VAL[0]}~{_GP_VAL[1]}  (test = 竞赛 live)")
    data = prepare_gp_data(_GP_TRAIN[0], _GP_TRAIN[1], _GP_VAL[0], _GP_VAL[1], fm.qnt_data_dir)
    seed = int.from_bytes(os.urandom(4), 'little')
    log(f"[alphasage] seed={seed}(每次运行随机指定;复现固定此 seed)")
    pool, _ = train_alphasage(
        data['train_panels'], capacity=asc.pool_capacity,
        hidden_dim=asc.hidden_dim, n_layers=asc.n_layers,
        lr=asc.lr, batch_size=asc.batch_size, n_episodes=asc.n_episodes,
        nov_weight=asc.nov_weight, entropy_coef=asc.entropy_coef,
        entropy_temperature=asc.entropy_temperature,
        r_sa_weight=asc.r_sa_weight, r_sa_k=asc.r_sa_k, r_sa_tau=asc.r_sa_tau,
        logz_lr=asc.logz_lr, seed=seed, device=_EVAL_DEVICE, log=log)
    v = eval_pool_val(pool, data, bcfg)
    log(f"[alphasage] pool_size={pool.size} val sharpe={v['sharpe']:+.4f}"
        f"(EW 锚 {v['ew_sharpe']:+.4f})maxDD={v['maxdd']:+.3f} ret={v['ret']:+.4f}")
    run_ts = os.environ.get('RUN_TS') or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    members, battery = _battery_filter(pool, data, log, 'alphasage')
    sub = submission_gate(members, data, bcfg, 'alphasage', run_ts)
    passed_trees = [t for (t, _), r in zip(members, sub) if r['passed']]
    if passed_trees:
        sb = _val_book(passed_trees, data, bcfg)
        _ledger_append({'ts': _now_iso(), 'engine': 'alphasage', 'run': run_ts,
                        'kind': 'submission_book', 'n_members': len(passed_trees),
                        'val_sharpe': round(sb['sharpe'], 4), 'ew_val': round(sb['ew_sharpe'], 4)})
        log(f"[alphasage] 二级信号门 {len(passed_trees)}/{len(sub)} 过门  提交书 "
            f"val sharpe={sb['sharpe']:+.4f}(EW 锚 {sb['ew_sharpe']:+.4f})"
            f"maxDD={sb['maxdd']:+.3f}(门后读数,选择偏差已入)")
    else:
        log(f"[alphasage] 二级信号门 0/{len(sub)} 过门(双窗复制门空手=正常预期)")
    snap = Path(f'logs/alphasage_pool_{run_ts}.jsonl')
    snap.parent.mkdir(parents=True, exist_ok=True)
    with open(snap, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'pool': pool.to_jsonable(), 'val': v, 'submission': sub,
                            'battery': battery, 'seed': seed}, ensure_ascii=False) + '\n')
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({'pool': pool.to_jsonable(),
                   'submission': [r for r in sub if r['passed']], 'seed': seed},
                  f, ensure_ascii=False, indent=2)   # {'pool':[...]} = 部署/materialize 同契约
    log(f"[alphasage] deploy bundle → {output};snapshot → {snap}")


# ============================================================================
# 入口
# ============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(prog='quant', description='公式化 alpha 因子挖掘')
    sub = p.add_subparsers(dest='cmd', required=True)

    pg = sub.add_parser('gp-baseline', help='DEAP 强类型 GP 因子挖掘(NSGA-II;产 alpha pool JSON)')
    pg.add_argument('output', help='deploy bundle JSON 路径')

    pas = sub.add_parser('alphasage', help='AlphaSAGE GFlowNet 因子挖掘(独立 baseline;产 alpha pool JSON)')
    pas.add_argument('output', help='deploy bundle JSON 路径')

    args = p.parse_args(argv)
    if args.cmd == 'gp-baseline':
        cmd_gp_baseline(args.output)
    else:
        cmd_alphasage(args.output)


if __name__ == '__main__':
    main()

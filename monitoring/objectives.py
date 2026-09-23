"""What a search objective puts on the dashboard: funnel stages, verdicts, candidate columns, footnotes.

The run writes its objective into the status ([search.portfolio] --objective). Components ask the
objective object and never look at objective-specific report fields themselves, so a new objective
is one subclass registered in OBJECTIVES - no component or layout code changes.
"""
from typing import Callable, NamedTuple

from .cells import ACCENT, GREEN, INK2, INK3, INK4, AMBER, RED, Seg, num, tone


class Stage(NamedTuple):
    """One step of the candidate funnel; each stage counts a subset of the stage above it."""
    label: str
    key: str                     # cumulative counter in the status
    color: str
    blocked: str = ''            # why nothing got through; {n:,} = how many reached this stage


class Column(NamedTuple):
    """One column of the recent-candidates table, to the right of the expression and the verdict."""
    name: str
    width: int
    cell: Callable               # (report, status, background) -> [Seg]
    min_inner: int = 0           # a narrower card drops the column


def value(key, spec):
    return lambda report, st, bg: [Seg(num(report.get(key), spec), INK2, bg)]


def delta(key, spec, scale=1, suffix=''):
    def cell(report, st, bg):
        moved = (report.get('delta') or {}).get(key)
        return [Seg(num(None if moved is None else moved*scale, spec, suffix), tone(moved), bg)]
    return cell


class Objective:
    key = label = caption = ''
    passed_label = gate_key = ''             # the last numbered gate: its name on the tile and its status counter
    columns = ()

    def stages(self, st):
        raise NotImplementedError

    def rejected(self, report):
        """(label, colour) of a sound candidate that was not admitted."""
        raise NotImplementedError

    def footnotes(self, st):
        """(label, text) under the funnel, most important first; a narrow card drops them from the end."""
        raise NotImplementedError

    def counters(self):
        return tuple(stage.key for stage in self.stages({}))

    def verdict(self, report):
        if report.get('passed'):
            return '可接纳', GREEN
        if not report.get('g1'):
            return 'G1 未过', RED
        return self.rejected(report)

    def diagnosis(self, st):
        """(bottleneck stage key, tag, colour, message) - why the search has or has not produced."""
        if st.get('phase') == 'failed':
            return None, '失败', RED, st.get('error') or '运行失败，请查看任务日志'
        if not st.get('unique_evaluations'):
            return None, '等待', INK3, st.get('error') or '等待首批评价数据'
        # A counter the run does not report (a status older than the stage) is no evidence of a bottleneck.
        stages = [stage for stage in self.stages(st)
                  if (stage.blocked or stage.key == 'unique_evaluations') and st.get(stage.key) is not None]
        for above, stage in zip(stages, stages[1:]):
            if not st[stage.key]:
                return stage.key, '瓶颈', AMBER, f'{stage.label} — ' + stage.blocked.format(n=st.get(above.key, 0))
        return None, '产出', GREEN, f"已接纳 {st.get('accepted', 0):,} 个构件 · {st.get(stages[-1].key, 0):,} 个候选通过全部检查"


class Portfolio(Objective):
    """Reward = what the candidate adds to the portfolio when it is tried in it."""
    key, label, passed_label, gate_key = 'portfolio', '组合目标', 'G2 通过', 'g2'
    caption = '最新在前 · Δ 为候选试配后的组合变化'
    columns = (Column('ρ', 7, value('rho', '.3f')), Column('权重', 6, value('weight', '.0%'), 104),
               Column('Δ Sharpe', 10, delta('sharpe', '+.4f'), 84), Column('Δ CAGR', 11, delta('cagr', '+.2f', 100, ' pp')),
               Column('相似度', 7, value('novelty', '.2f'), 104))

    def stages(self, st):
        return (Stage('候选请求', 'requested', INK4), Stage('去重评价', 'unique_evaluations', ACCENT),
                Stage('G1 结构·因果', 'g1', ACCENT, '{n:,} 个候选尚无一通过'),
                Stage('G2 组合收益', 'g2', ACCENT, '{n:,} 个有效候选尚无一能改善组合'),
                Stage('可接纳', 'low_corr', GREEN, '{n:,} 个候选改善组合，但都与现有成员重复'))

    def rejected(self, report):
        if not report.get('g2'):
            return 'G2 未过', AMBER
        return ('行为重复', INK3) if report.get('duplicate') else ('相关未过', AMBER)

    def footnotes(self, st):
        return [(label, num(st.get(key))) for label, key in
                (('行为重复', 'behavior_duplicates'), ('FP64 精算', 'fp64_refined'), ('探针评价', 'probe_evaluations'))]


def sharpe_cell(report, st, bg):
    sharpe, needed = report.get('sharpe'), report.get('required_sharpe')
    return [Seg(num(sharpe, '.2f'), GREEN if None not in (sharpe, needed) and sharpe >= needed else INK2, bg)]


def years_cell(report, st, bg):
    years, floor = report.get('recent_years'), (st.get('gate') or {}).get('recent_year_sharpe')
    if not years:
        return [Seg('—', INK2, bg)]
    return [Seg('/'.join(f'{y:+.2f}' for y in years), GREEN if floor is not None and min(years) > floor else INK2, bg)]


class Single(Objective):
    """Reward = the candidate's own book (search/portfolio/single.py). Gates: G1 sound causal book, G2 appraisal
    ratio against the portfolio >= kappa (the (correlation, Sharpe) gate), G3 every recent calendar year as well."""
    key, label, passed_label, gate_key = 'single', '单 alpha 目标', 'G3 通过', 'g3'
    caption = '最新在前 · 指标为候选单独回测 · G2 门槛随它与组合的相关性升降'
    columns = (Column('Sharpe', 7, sharpe_cell), Column('G2 门槛', 8, value('required_sharpe', '.2f')),
               Column('ρ 成员', 7, value('rho_members', '+.2f'), 84), Column('ρ 组合', 7, value('rho_portfolio', '+.2f'), 104),
               Column('G3 近年', 12, years_cell),
               Column('maxDD', 7, value('maxdd', '.1%'), 84), Column('CAGR', 8, value('cagr', '.1%'), 120),
               Column('效用', 6, value('utility', '.3f'), 120))
    G2, G3 = 'g2', 'g3'                      # status counters and report flags of the two gates

    def stages(self, st):
        gate = st.get('gate') or {}
        kappa, years, floor = (gate.get(k) for k in ('required_appraisal', 'recent_years', 'recent_year_sharpe'))
        appraisal = 'G2 门已关' if gate and kappa is None else 'G2 Sharpe(ρ)'
        recent = 'G3 近年 SR' if floor is None else f'G3 近{years}年 SR>{floor:g}'
        return (Stage('去重评价', 'unique_evaluations', ACCENT),
                Stage('G1 结构·因果', 'g1', ACCENT, '{n:,} 个候选尚无一通过'),
                Stage(appraisal, self.G2, ACCENT, '{n:,} 个有效候选的 Sharpe 都低于各自相关性对应的门槛'),
                Stage(recent, self.G3, ACCENT, '{n:,} 个候选过了 G2，但近年没有做到每年 Sharpe 达标'),
                Stage('可接纳', 'low_corr', GREEN, '{n:,} 个候选过了 G3，但都与现有成员重复'))

    def rejected(self, report):
        if not report.get('g2'):
            return 'G2 未过', AMBER
        if not report.get('g3'):
            return 'G3 未过', AMBER
        return ('行为重复', INK3) if report.get('duplicate') else ('相关未过', AMBER)

    def footnotes(self, st):
        gate = st.get('gate') or {}
        anchor = [('锚', f"ρ {gate['anchor_correlation']:g} ↔ Sharpe {gate['anchor_sharpe']:g}")] if 'anchor_sharpe' in gate else []
        return anchor + [('κ', num(gate.get('required_appraisal'), '.3f')), ('先验均值', num(gate.get('prior_mean'), '+.3f')),
                         ('收缩', num(gate.get('shrinkage'), '.3f')), ('行为重复', num(st.get('behavior_duplicates'))),
                         ('探针评价', num(st.get('probe_evaluations')))]


class SingleUnnumbered(Single):
    """single_0921 only, the run before the gates were numbered: its g2 flag and counter mean the whole gate
    (today's G3) and it did not count the appraisal gate alone, so G2 shows no count."""
    G2, G3, gate_key = 'g2_unreported', 'g2', 'g2'

    def rejected(self, report):
        if report.get('g2'):
            return ('行为重复', INK3) if report.get('duplicate') else ('相关未过', AMBER)
        sharpe, needed = report.get('sharpe'), report.get('required_sharpe')
        return ('G2 未过', AMBER) if needed is None or sharpe < needed else ('G3 未过', AMBER)


OBJECTIVES = {objective.key: objective for objective in (Portfolio(), Single())}
UNNUMBERED = SingleUnnumbered()


def of(st):
    """A status names its objective. The ones written before that (single_0921 and older) are told apart by
    the gate: only the single-alpha objective has one, the portfolio objective writes null."""
    if 'objective' in st:
        return OBJECTIVES[st['objective']]
    return UNNUMBERED if st.get('gate') else OBJECTIVES['portfolio']

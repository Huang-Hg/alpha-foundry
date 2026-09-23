"""Invariants of the terminal dashboard (foundry/monitor): every frame exactly fills the window, the HTML preview and
the terminal draw the same cells, blocks give way as the window shrinks, one frame on a pipe, the alternate screen,
readings glide, a new objective is one registered object, and every status built here is declared in schema.py."""
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import UnionType
from typing import Annotated, Union, get_args, get_origin, get_type_hints, is_typeddict
import unittest
from unittest.mock import Mock, patch

from rich.cells import cell_len
from rich.console import Console

from foundry.monitor import objectives
from foundry.monitor.preview import export
from foundry.monitor.schema import GLIDE, RESTART, SearchStatus, ServiceStatus
from foundry.monitor.view import compose, render
from foundry.monitor.watch import Glide, watch

GRID = ((200, 60), (120, 30), (96, 24), (80, 24), (72, 18), (60, 15), (56, 12), (56, 8))

# Every status the tests draw is built here, so the schema test covers all of them.
COMPLETE = dict(phase='complete', epoch=5, epochs=5, accepted=0, unique_evaluations=1279, g1=1193, g2=0,
                portfolio_sharpe=2.275, portfolio_cagr=1.2012,
                recent=[dict(expr='long_expression_'*30, g1=True, g2=False, rho=.2, delta={'cagr': -.09})]*4)
RUNNING = dict(phase='evaluating', epoch=3, epochs=5, elapsed_seconds=4000, requested=1580,
               unique_evaluations=1279, g1=1193, g2=6, low_corr=0,
               recent=[dict(expr='ts_slope(ts_ref(gt(中文字段,%d),120),20)' % i*4, g1=True, g2=i % 2 == 0, passed=False,
                            rho=.2, weight=.05, novelty=.3, delta={'sharpe': .0001, 'cagr': -.09}) for i in range(6)],
               sampling_seconds=.02, evaluation_seconds=7.5, training_seconds=.08, reweight_seconds=0.,
               loss_before=431.6, loss_after=416.1, device='Tesla V100S-PCIE-32GB', mode='gfn')
FILLED = [{}, dict(phase='failed', error='boom '*80, epoch=2, epochs=3), RUNNING]

GATE = dict(prior_mean=-.474, shrinkage=.462, reference_sharpe=1.08, required_appraisal=.871,
            anchor_correlation=.7, anchor_sharpe=1.5, recent_years=2, recent_year_sharpe=1.)
REPORT = dict(expr='ts_max(log(add(abs(lt(gt(btc_feepertx,amihud),ma_dist_120)),candlebody)),7)', g1=True, rho=.4,
              novelty=.3, weight=1., utility=.423, sharpe=1.22, cagr=.172, maxdd=-.178, rho_portfolio=.4,
              appraisal=.88, required_sharpe=1.12, delta=dict(sharpe=.1, cagr=.172))
SINGLE = dict(phase='evaluating', objective='single', mode='gfn', epoch=3, epochs=5, gate=GATE, requested=1580,
              unique_evaluations=1279, g1=1193, g2=52, g3=0, low_corr=0, accepted=0,
              recent=[dict(REPORT, g2=True, g3=True, passed=True, recent_years=[1.07, 1.58]),
                      dict(REPORT, g2=True, g3=False, passed=False, recent_years=[-.7, .94]),
                      dict(REPORT, g2=False, g3=False, passed=False, sharpe=.9, recent_years=[1.2, 1.3]),
                      dict(expr='div(close,0)', g1=False, g2=False, g3=False, passed=False)])
SINGLE_AS_PORTFOLIO = dict(SINGLE, objective='portfolio')
SINGLE_CLOSED = dict(SINGLE, gate=dict(GATE, required_appraisal=None))
PROBE = dict(phase='evaluating', objective='probe', mode='gfn', epoch=1, epochs=2, unique_evaluations=5,
             g1=5, g2=1, low_corr=1, recent=[dict(expr='close', g1=True, g2=True, passed=True)])
MARKET = dict(RUNNING, market='crypto')

NOW = 1000.
JOB = dict(t=NOW-40, op='family', source='tmp/funnel/agents/r7/vol_regime/round2.txt', dt=12.3, ok=True,
           n=24, variants=9311, verdicts={'无效': 21, '成立': 2, '稀疏': 1})
BUSY = dict(kind='service', phase='serve', now=NOW, elapsed_seconds=4000, busy_seconds=1000, queue=2, clients=3,
            served=9, exprs=48213, batches=9, vram='5120/8192MiB', device='Tesla V100S-PCIE-32GB · combo auto',
            portfolio=dict(n=4, sharpe=2.277, cagr=1.428, maxdd=-.369),
            cur=dict(op='family', n=17, t0=NOW-75, clients=1, source='tmp\\funnel\\agents\\r7\\breadth\\round3.txt',
                     stage='GPU 邻域展开', done=1024, total=4096),
            jobs=[dict(JOB, t=NOW-300+i, ok=i != 17) for i in range(20)],
            sources={'vol_regime': dict(rounds=2, formulas=40, variants=18000, verdicts={'无效': 38, '成立': 2}, last=NOW-40),
                     'breadth': dict(rounds=1, formulas=12, variants=2000, verdicts={'无效': 12}, last=NOW-900)})
IDLE = dict(BUSY, cur=None)
SERVICE = [dict(kind='service', phase='warm', warm='装载面板与派生特征…', now=NOW, elapsed_seconds=5),
           IDLE, BUSY, dict(BUSY, _source_mtime=0., _source_name='service_status.json')]
SERVICE_STALE = dict(BUSY, _source_mtime=time.time()-120)
SERVICE_MARKET = dict(BUSY, market='crypto')

GLIDES = [dict(unique_evaluations=100, portfolio_sharpe=2.0, phase='evaluating'),
          dict(unique_evaluations=300, portfolio_sharpe=2.2, phase='evaluating'),
          dict(unique_evaluations=7, phase='loading')]
FILES = [dict(phase='waiting'), dict(unique_evaluations=111), dict(unique_evaluations=222)]

SEARCH_STATES = [COMPLETE, *FILLED, SINGLE, SINGLE_AS_PORTFOLIO, SINGLE_CLOSED, PROBE, MARKET, *GLIDES, *FILES]
SERVICE_STATES = [*SERVICE, SERVICE_STALE, SERVICE_MARKET]


def undeclared(value, hint, path=''):
    """Keys of `value`, at any depth, that the schema type `hint` does not declare."""
    if get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    if get_origin(hint) in (Union, UnionType):
        if value is None:
            return []
        (hint,) = [arg for arg in get_args(hint) if arg is not type(None)]
    if is_typeddict(hint):
        fields = get_type_hints(hint, include_extras=True)
        return [missing for key, item in value.items()
                for missing in ([path+key] if key not in fields else undeclared(item, fields[key], path+key+'.'))]
    if get_origin(hint) is list:
        return [missing for item in value for missing in undeclared(item, get_args(hint)[0], path)]
    if get_origin(hint) is dict:
        return [missing for item in value.values() for missing in undeclared(item, get_args(hint)[1], path)]
    return []


class Terminal(io.StringIO):
    def isatty(self):
        return True


class MonitorTests(unittest.TestCase):
    def assertFills(self, state, width, height):
        frame = compose(state, width, height, tick=3, interval=.5)
        self.assertEqual(len(frame), height)
        self.assertEqual({sum(cell_len(s.text) for s in row) for row in frame}, {width})
        return frame

    def test_layout_fits_and_keeps_results_visible_at_small_sizes(self):
        for width, height in ((120, 32), (100, 30), (80, 25), (70, 20), (60, 15)):
            with self.subTest(width=width, height=height):
                output = io.StringIO()
                console = Console(file=output, width=width, height=height)
                console.print(render(COMPLETE, rows=height, width=width))
                rendered = output.getvalue()
                self.assertLessEqual(len(rendered.splitlines()), height)
                self.assertIn('2.275', rendered)
                self.assertIn('120.12%', rendered)
                self.assertIn('1,279', rendered)
                self.assertIn('新增构件', rendered)

    def test_every_row_fills_the_grid_and_both_backends_draw_the_same_cells(self):
        for state in FILLED:
            for width, height in GRID:
                with self.subTest(phase=state.get('phase'), width=width, height=height):
                    frame = self.assertFills(state, width, height)
                    plain = render(state, rows=height, width=width, tick=3, interval=.5).plain.split('\n')
                    self.assertEqual(plain, [''.join(s.text for s in row) for row in frame])
                    self.assertEqual(export(frame, {}).count('<div class=r>'), height)

    def test_single_alpha_run_shows_its_own_gate_columns_and_verdicts(self):
        for width, height in ((200, 60), (120, 40), (96, 30), (80, 24), (60, 15), (56, 12), (56, 8)):
            with self.subTest(width=width, height=height):
                self.assertFills(SINGLE, width, height)
        screen = render(SINGLE, rows=40, width=120).plain
        for text in ('单 alpha 目标', 'G1 结构·因果', 'G2 Sharpe(ρ)', 'G3 近2年 SR>1', '锚 ρ 0.7 ↔ Sharpe 1.5', 'κ 0.871', '先验均值 -0.474',
                     '可接纳', 'G3 未过', 'G2 未过', 'G1 未过', 'G3 通过 0', '+1.07/+1.58', '-17.8%', '52 个候选过了 G2'):
            self.assertIn(text, screen)
        for text in ('Δ Sharpe', 'Δ CAGR', '相似度', 'G2 组合收益'):
            self.assertNotIn(text, screen)
        # The same reports under the portfolio objective keep the portfolio columns; a closed gate says so.
        self.assertIn('Δ Sharpe', render(SINGLE_AS_PORTFOLIO, rows=40, width=120).plain)
        self.assertIn('G2 门已关', render(SINGLE_CLOSED, rows=40, width=120).plain)

    def test_evaluation_service_screen_follows_jobs_and_submitters(self):
        for state in SERVICE:
            for width, height in GRID:
                with self.subTest(phase=state['phase'], cur=bool(state.get('cur')), width=width, height=height):
                    self.assertFills(state, width, height)
        screen = render(BUSY, rows=40, width=140).plain
        for text in ('评价服务', 'V100S 32GB', '计算中', '假设族', 'breadth', 'round3.txt', 'GPU 邻域展开', '1,024/4,096',
                     '2.277', '48,213', '25%', '提交者', 'vol_regime', '18,000', '成立 2', '最近作业', '稀疏 1', '失败'):
            self.assertIn(text, screen)
        self.assertLess(screen.index('vol_regime'), screen.index('breadth  '))       # most recent submitter first
        self.assertIn('空闲', render(IDLE, rows=40, width=140).plain)
        self.assertIn('数据停滞', render(SERVICE_STALE, rows=40, width=140).plain)

    def test_title_names_the_market_only_when_the_status_does(self):
        header = lambda state: render(state, rows=30, width=120).plain.split('\n')[0]
        self.assertNotIn('ALPHA SEARCH · ', header(RUNNING))
        self.assertNotIn('评价服务 · ', header(BUSY))
        self.assertIn('ALPHA SEARCH · CRYPTO', header(MARKET))
        self.assertIn('评价服务 · CRYPTO', header(SERVICE_MARKET))
        for state in (MARKET, SERVICE_MARKET):
            for width, height in GRID:
                with self.subTest(kind=state.get('kind'), width=width, height=height):
                    self.assertNotIn('Q25', ''.join(s.text for s in self.assertFills(state, width, height)[0]))

    def test_a_new_objective_is_one_registered_object(self):
        class Probe(objectives.Portfolio):
            key, label = 'probe', '探针目标'
            columns = (objectives.Column('自定义列', 10, lambda report, st, bg: [objectives.Seg('42', bg=bg)]),)

        with patch.dict(objectives.OBJECTIVES, {'probe': Probe()}):
            screen = render(PROBE, rows=40, width=120).plain
        self.assertIn('探针目标', screen)
        self.assertIn('自定义列', screen)
        self.assertIn('42', screen)

    def test_pipe_prints_one_snapshot_and_does_not_claim_live_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            path.write_text(json.dumps(FILES[0]), encoding='utf-8')
            output = io.StringIO()
            heartbeat = Mock()
            with patch('sys.stdout', output), patch('foundry.monitor.watch.time.sleep') as sleep:
                watch(path, heartbeat=heartbeat)
            self.assertEqual(output.getvalue().count('ALPHA SEARCH'), 1)
            sleep.assert_not_called()
            heartbeat.assert_not_called()

    def test_live_updates_one_screen_and_restores_it_on_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            source = Path(directory) / 'source.json'
            source.write_text(json.dumps({'status_path': str(path)}), encoding='utf-8')
            path.write_text(json.dumps(FILES[1]), encoding='utf-8')
            output = Terminal()
            frames, clock = [], [0.]

            def advance(seconds):
                clock[0] += seconds
                frames.append(output.getvalue())
                if len(frames) == 1:
                    path.write_text(json.dumps(FILES[2]), encoding='utf-8')
                if len(frames) == 30:
                    raise KeyboardInterrupt

            term = 'dumb' if os.name == 'nt' else 'xterm-256color'
            with patch.dict(os.environ, {'TERM': term, 'NO_COLOR': '1'}):
                with patch('sys.stdout', output), patch('foundry.monitor.watch.time.sleep', advance), \
                        patch('foundry.monitor.watch.time.monotonic', lambda: clock[0]):
                    with self.assertRaises(KeyboardInterrupt):
                        watch(source)
            stream = output.getvalue()
            self.assertIn('111', frames[0])
            self.assertIn('222', stream[len(frames[0]):])
            self.assertEqual(stream.count('\x1b[?1049h'), 1)
            self.assertEqual(stream.count('\x1b[?1049l'), 1)
            self.assertIn('\x1b[H', stream)
            self.assertIn('\x1b[?25h', stream)

    def test_readings_glide_to_the_reported_value_and_never_past_it(self):
        glide = Glide()
        glide.feed(GLIDES[0], 0.)
        self.assertEqual(glide.at(0.)['unique_evaluations'], 100)        # first snapshot: shown at once
        glide.feed(GLIDES[1], 2.)
        self.assertEqual(glide.at(2.)['unique_evaluations'], 100)
        self.assertEqual(glide.at(3.)['unique_evaluations'], 200)        # halfway through a 2 s gap
        self.assertAlmostEqual(glide.at(3.)['portfolio_sharpe'], 2.1)
        self.assertEqual(glide.at(9.)['unique_evaluations'], 300)        # holds at the report, no extrapolation
        glide.feed(GLIDES[2], 10.)
        self.assertEqual(glide.at(10.)['unique_evaluations'], 7)         # a new run does not count down
        # Every objective's funnel counters glide; the run and service lifetimes tell a restart.
        stages = {stage.key for objective in objectives.OBJECTIVES.values() for stage in objective.stages({})}
        self.assertLessEqual(stages, set(GLIDE))
        self.assertEqual(set(RESTART), {'epoch', 'unique_evaluations', 'exprs'})

    def test_schema_declares_every_field_the_tests_build(self):
        for schema, states in ((SearchStatus, SEARCH_STATES), (ServiceStatus, SERVICE_STATES)):
            for i, state in enumerate(states):
                with self.subTest(schema=schema.__name__, state=i):
                    self.assertEqual(undeclared(state, schema), [])
        # The walker itself must see a drifted key, top-level and nested.
        self.assertEqual(undeclared(dict(RUNNING, drift=1), SearchStatus), ['drift'])
        self.assertEqual(undeclared(dict(BUSY, cur=dict(BUSY['cur'], drift=1)), ServiceStatus), ['cur.drift'])


if __name__ == '__main__':
    unittest.main()

"""Regression checks for persistent redraw and redirected output."""
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from rich.console import Console

from rich.cells import cell_len

from .preview import export
from .q25 import Glide, render, watch
from .view import compose


class Terminal(io.StringIO):
    def isatty(self):
        return True


class WatchTests(unittest.TestCase):
    def test_layout_fits_and_keeps_results_visible_at_small_sizes(self):
        state = dict(phase='complete', epoch=5, epochs=5, accepted=0,
                     unique_evaluations=1279, g1=1193, g2=0,
                     portfolio_sharpe=2.275, portfolio_cagr=1.2012,
                     recent=[dict(expr='long_expression_'*30, g1=True, g2=False,
                                  rho=.2, delta={'cagr':-.09})]*4)
        for width, height in ((120,32),(100,30),(80,25),(70,20),(60,15)):
            with self.subTest(width=width, height=height):
                output = io.StringIO()
                console = Console(file=output, width=width, height=height)
                console.print(render(state, rows=height, width=width))
                rendered = output.getvalue()
                self.assertLessEqual(len(rendered.splitlines()), height)
                self.assertIn('2.275', rendered)
                self.assertIn('120.12%', rendered)
                self.assertIn('1,279', rendered)
                self.assertIn('新增构件', rendered)

    def test_every_row_fills_the_grid_and_both_backends_draw_the_same_cells(self):
        recent = [dict(expr='ts_slope(ts_ref(gt(中文字段,%d),120),20)' % i*4, g1=True, g2=i % 2 == 0, passed=False,
                       rho=.2, weight=.05, novelty=.3, delta={'sharpe': .0001, 'cagr': -.09}) for i in range(6)]
        states = [{}, dict(phase='failed', error='boom '*80, epoch=2, epochs=3),
                  dict(phase='evaluating', epoch=3, epochs=5, elapsed_seconds=4000, requested=1580,
                       unique_evaluations=1279, g1=1193, g2=6, low_corr=0, recent=recent,
                       sampling_seconds=.02, evaluation_seconds=7.5, training_seconds=.08, reweight_seconds=0.,
                       loss_before=431.6, loss_after=416.1, device='Tesla V100S-PCIE-32GB', mode='gfn')]
        for state in states:
            for width, height in ((200,60),(120,30),(96,24),(80,24),(72,18),(60,15),(56,12),(56,8)):
                with self.subTest(phase=state.get('phase'), width=width, height=height):
                    frame = compose(state, width, height, tick=3, interval=.5)
                    self.assertEqual(len(frame), height)
                    self.assertEqual({sum(cell_len(s.text) for s in row) for row in frame}, {width})
                    plain = render(state, rows=height, width=width, tick=3, interval=.5).plain.split('\n')
                    self.assertEqual(plain, [''.join(s.text for s in row) for row in frame])
                    self.assertEqual(export(frame, {}).count('<div class=r>'), height)

    def test_single_alpha_run_shows_its_own_gate_columns_and_verdicts(self):
        gate = dict(prior_mean=-.474, shrinkage=.462, reference_sharpe=1.08, required_appraisal=.871,
                    anchor_correlation=.7, anchor_sharpe=1.5, recent_years=2, recent_year_sharpe=1.)
        report = dict(expr='ts_max(log(add(abs(lt(gt(btc_feepertx,amihud),ma_dist_120)),candlebody)),7)', g1=True, rho=.4,
                      novelty=.3, weight=1., utility=.423, sharpe=1.22, cagr=.172, maxdd=-.178, rho_portfolio=.4,
                      appraisal=.88, required_sharpe=1.12, delta=dict(sharpe=.1, cagr=.172))
        recent = [dict(report, g2=True, g3=True, passed=True, recent_years=[1.07, 1.58]),
                  dict(report, g2=True, g3=False, passed=False, recent_years=[-.7, .94]),
                  dict(report, g2=False, g3=False, passed=False, sharpe=.9, recent_years=[1.2, 1.3]),
                  dict(expr='div(close,0)', g1=False, g2=False, g3=False, passed=False)]
        state = dict(phase='evaluating', objective='single', mode='gfn', epoch=3, epochs=5, gate=gate, requested=1580,
                     unique_evaluations=1279, g1=1193, g2=52, g3=0, low_corr=0, accepted=0, recent=recent)
        for width, height in ((200,60),(120,40),(96,30),(80,24),(60,15),(56,12),(56,8)):
            with self.subTest(width=width, height=height):
                frame = compose(state, width, height, tick=3, interval=.5)
                self.assertEqual(len(frame), height)
                self.assertEqual({sum(cell_len(s.text) for s in row) for row in frame}, {width})
        screen = render(state, rows=40, width=120).plain
        for text in ('单 alpha 目标', 'G1 结构·因果', 'G2 Sharpe(ρ)', 'G3 近2年 SR>1', '锚 ρ 0.7 ↔ Sharpe 1.5', 'κ 0.871', '先验均值 -0.474',
                     '可接纳', 'G3 未过', 'G2 未过', 'G1 未过', 'G3 通过 0', '+1.07/+1.58', '-17.8%', '52 个候选过了 G2'):
            self.assertIn(text, screen)
        for text in ('Δ Sharpe', 'Δ CAGR', '相似度', 'G2 组合收益'):
            self.assertNotIn(text, screen)
        # The same reports under the portfolio objective keep the portfolio columns; a closed gate says so.
        self.assertIn('Δ Sharpe', render(dict(state, objective='portfolio'), rows=40, width=120).plain)
        self.assertIn('G2 门已关', render(dict(state, gate=dict(gate, required_appraisal=None)), rows=40, width=120).plain)
        # single_0921 predates the numbering: its g2 is today's G3, and the appraisal gate alone was not counted.
        old = {k: v for k, v in state.items() if k not in ('objective', 'g3')}
        old.update(g2=2, low_corr=2, recent=[dict(r, g2=r.get('g3', False)) for r in recent])
        screen = render(old, rows=40, width=120).plain
        for text in ('单 alpha 目标', 'G3 通过 2', 'G2 未过', 'G3 未过'):
            self.assertIn(text, screen)
        self.assertRegex(screen, r'G2 Sharpe\(ρ\) +─+ +— ')

    def test_evaluation_service_screen_follows_jobs_and_submitters(self):
        now = 1000.
        job = dict(t=now-40, op='family', source='tmp/q25_funnel/agents/r7/vol_regime/round2.txt', dt=12.3, ok=True,
                   n=24, variants=9311, verdicts={'无效': 21, '成立': 2, '稀疏': 1})
        busy = dict(kind='service', phase='serve', now=now, elapsed_seconds=4000, busy_seconds=1000, queue=2, clients=3,
                    served=9, exprs=48213, batches=9, vram='5120/8192MiB', device='Tesla V100S-PCIE-32GB · combo auto',
                    portfolio=dict(n=4, sharpe=2.277, cagr=1.428, maxdd=-.369),
                    cur=dict(op='family', n=17, t0=now-75, clients=1, source='tmp\\q25_funnel\\agents\\r7\\breadth\\round3.txt',
                             stage='GPU 邻域展开', done=1024, total=4096),
                    jobs=[dict(job, t=now-300+i, ok=i != 17) for i in range(20)],
                    sources={'vol_regime': dict(rounds=2, formulas=40, variants=18000, verdicts={'无效': 38, '成立': 2}, last=now-40),
                             'breadth': dict(rounds=1, formulas=12, variants=2000, verdicts={'无效': 12}, last=now-900)})
        states = [dict(kind='service', phase='warm', warm='装载面板与派生特征…', now=now, elapsed_seconds=5),
                  dict(busy, cur=None), busy, dict(busy, _source_mtime=0., _source_name='service_status.json')]
        for state in states:
            for width, height in ((200,60),(120,30),(96,24),(80,24),(72,18),(60,15),(56,12),(56,8)):
                with self.subTest(phase=state['phase'], cur=bool(state.get('cur')), width=width, height=height):
                    frame = compose(state, width, height, tick=3, interval=.5)
                    self.assertEqual(len(frame), height)
                    self.assertEqual({sum(cell_len(s.text) for s in row) for row in frame}, {width})
        screen = render(busy, rows=40, width=140).plain
        for text in ('Q25 评价服务', 'V100S 32GB', '计算中', '假设族', 'breadth', 'round3.txt', 'GPU 邻域展开', '1,024/4,096',
                     '2.277', '48,213', '25%', '提交者', 'vol_regime', '18,000', '成立 2', '最近作业', '稀疏 1', '失败'):
            self.assertIn(text, screen)
        self.assertLess(screen.index('vol_regime'), screen.index('breadth  '))       # most recent submitter first
        self.assertIn('空闲', render(dict(busy, cur=None), rows=40, width=140).plain)
        self.assertIn('数据停滞', render(dict(busy, _source_mtime=time.time()-120), rows=40, width=140).plain)

    def test_a_new_objective_is_one_registered_object(self):
        from . import objectives

        class Probe(objectives.Portfolio):
            key, label = 'probe', '探针目标'
            columns = (objectives.Column('自定义列', 10, lambda report, st, bg: [objectives.Seg('42', bg=bg)]),)

        with patch.dict(objectives.OBJECTIVES, {'probe': Probe()}):
            screen = render(dict(phase='evaluating', objective='probe', mode='gfn', epoch=1, epochs=2, unique_evaluations=5,
                                 g1=5, g2=1, low_corr=1, recent=[dict(expr='close', g1=True, g2=True, passed=True)]),
                            rows=40, width=120).plain
        self.assertIn('探针目标', screen)
        self.assertIn('自定义列', screen)
        self.assertIn('42', screen)

    def test_pipe_prints_one_snapshot_and_does_not_claim_live_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            path.write_text('{"phase":"waiting"}', encoding='utf-8')
            output = io.StringIO()
            heartbeat = Mock()
            with patch('sys.stdout', output), patch('monitoring.q25.time.sleep') as sleep:
                watch(path, heartbeat=heartbeat)
            self.assertEqual(output.getvalue().count('ALPHA SEARCH'), 1)
            sleep.assert_not_called()
            heartbeat.assert_not_called()

    def test_live_updates_one_screen_and_restores_it_on_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            source = Path(directory) / 'source.json'
            source.write_text(json.dumps({'status_path': str(path)}), encoding='utf-8')
            path.write_text('{"unique_evaluations":111}', encoding='utf-8')
            output = Terminal()
            frames, clock = [], [0.]

            def advance(seconds):
                clock[0] += seconds
                frames.append(output.getvalue())
                if len(frames) == 1:
                    path.write_text('{"unique_evaluations":222}', encoding='utf-8')
                if len(frames) == 30:
                    raise KeyboardInterrupt

            term = 'dumb' if os.name == 'nt' else 'xterm-256color'
            with patch.dict(os.environ, {'TERM': term, 'NO_COLOR': '1'}):
                with patch('sys.stdout', output), patch('monitoring.q25.time.sleep', advance), \
                        patch('monitoring.q25.time.monotonic', lambda: clock[0]):
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
        glide.feed(dict(unique_evaluations=100, portfolio_sharpe=2.0, phase='evaluating'), 0.)
        self.assertEqual(glide.at(0.)['unique_evaluations'], 100)        # first snapshot: shown at once
        glide.feed(dict(unique_evaluations=300, portfolio_sharpe=2.2, phase='evaluating'), 2.)
        self.assertEqual(glide.at(2.)['unique_evaluations'], 100)
        self.assertEqual(glide.at(3.)['unique_evaluations'], 200)        # halfway through a 2 s gap
        self.assertAlmostEqual(glide.at(3.)['portfolio_sharpe'], 2.1)
        self.assertEqual(glide.at(9.)['unique_evaluations'], 300)        # holds at the report, no extrapolation
        glide.feed(dict(unique_evaluations=7, phase='loading'), 10.)
        self.assertEqual(glide.at(10.)['unique_evaluations'], 7)         # a new run does not count down


if __name__ == '__main__':
    unittest.main()

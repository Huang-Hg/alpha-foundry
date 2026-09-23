"""python -m foundry mine OUT: the search loop.

Per epoch the portfolio and the behaviour archive are frozen; the policy samples a batch, CUDA evaluates it against the
frozen portfolio, the policy trains on the rewards while the GPU admits the best candidate and reweights the members,
then the epoch is checkpointed. OUT holds status.json (mirrored to tmp/monitor/live.json for the dashboard),
epochs.jsonl, state.json and the policy checkpoints; `--resume` continues a run only under an identical signature
(protocol, configuration, source code, data, vocabulary, precision).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import cupy as cp

from foundry.config import ROOT, CudaConfig, ExprConfig, MarketConfig, SearchConfig, path
from foundry.core.files import publish
from foundry.engine import Engine
from foundry.ops import cuda
from foundry.portfolio.library import PROTOCOL, Library
from foundry.portfolio.portfolio import Portfolio
from foundry.portfolio.reference import CFG as PORTFOLIO
from foundry.search.evaluator import Archive, Evaluator
from foundry.search.policy import Policy

LIVE = ROOT / 'tmp' / 'monitor' / 'live.json'
SOURCES = ('.py', '.cpp', '.cu', '.hpp', '.h')
# Parts of foundry/ no result depends on: the dashboard, the evaluation service and the command line.
PRESENTATION = ('monitor', 'service', 'report.py', '__main__.py')


def write_json(path, value):
    publish(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
    if path.name == 'status.json':
        LIVE.parent.mkdir(parents=True, exist_ok=True)
        publish(LIVE, json.dumps(dict(value, run=str(path.parent)), ensure_ascii=False, allow_nan=False).encode('utf-8'))


def code_digest():
    """Every source that decides a result: engine, kernels, admission and search alike."""
    h = hashlib.sha256()
    package = ROOT / 'foundry'
    for p in sorted(p for p in package.rglob('*')
                    if p.suffix in SOURCES and p.relative_to(package).parts[0] not in PRESENTATION):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def train(policy, rewards, epoch, cfg):
    start = time.perf_counter()
    learned = policy.train(rewards, epoch, cfg.train_steps, cfg.learning_rate, cfg.opd_strength)
    return learned, time.perf_counter() - start


def run(args):
    if args.null_shift and args.seed_portfolio != 'equal':
        raise ValueError('The noise control delays every expression book, members included: start it from --seed-portfolio equal')
    if args.seed_portfolio == 'equal' and args.accept:
        raise ValueError('An equal-weight experiment cannot overwrite the production portfolio')
    if args.objective == 'single' and (args.accept or args.null_shift):
        raise ValueError('The single-alpha objective is an experiment: it cannot write the production portfolio, '
                         'and its gate needs undelayed candidates to measure against their delayed twins')
    cfg = replace(SearchConfig(), **{k: getattr(args, k) for k in ('epochs', 'batch_size', 'probes', 'seed') if getattr(args, k) is not None})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    if (output / 'state.json').exists() and not args.resume:
        raise ValueError('Output already has a run; use --resume or a new output directory')
    lock = output / '.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = time.perf_counter()
    # objective / null_shift tell the dashboard which funnel, verdicts and candidate columns this run has
    state = dict(phase='loading', market=MarketConfig().name, epoch=0, epochs=cfg.epochs, mode=args.mode,
                 objective=args.objective, null_shift=args.null_shift, accepted=0)
    write_json(status_path, state)
    trainer = ThreadPoolExecutor(max_workers=1)
    try:
        engine = Engine()
        if args.objective == 'single':
            from foundry.search import single
            Scorer, objective = single.SingleEvaluator, asdict(single.CFG)
        else:
            Scorer, objective = Evaluator, None
        signature = dict(protocol=PROTOCOL, config=asdict(cfg), expr=asdict(ExprConfig()), combo=asdict(PORTFOLIO),
                         mode=args.mode, seed_portfolio=args.seed_portfolio, null_shift=args.null_shift,
                         objective=args.objective, single=objective, code=code_digest(),
                         market=dict(asdict(engine.market.cfg), name=engine.market.name),
                         data=[(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                               for p in sorted(path(engine.market.cfg.data_dir).glob('*.parquet'))],
                         vocabulary=hashlib.sha256('\n'.join(engine.vocabulary).encode()).hexdigest(),
                         precision=dict(asdict(CudaConfig()), candidate_batch=engine.candidate_batch,
                                        trial_batch=engine.trial_batch))
        policy = Policy(engine.vocabulary, cfg.seed, cfg.max_tokens, ExprConfig().max_cost, cfg.max_ts_depth,
                        (cuda.MAX_WINDOW, cuda.MAX_PAIR_WINDOW))
        library = Library([]) if args.seed_portfolio == 'equal' else Library.load()
        portfolio = Portfolio(engine, library, null_shift=args.null_shift)
        archive = Archive(cfg.archive_size)
        begin = 0
        totals = dict.fromkeys(Scorer.COUNTERS + ('probe_evaluations', 'reweight_evaluations'), 0)
        if args.resume:
            old = json.loads((output / 'state.json').read_text(encoding='utf-8'))
            # Compare JSON forms so tuple/list differences do not break signatures.
            if json.loads(json.dumps(signature)) != old['signature']:
                raise ValueError('Resume data/config/protocol mismatch')
            if args.accept and old['members'] != library.members:
                raise ValueError('Production portfolio changed since checkpoint')
            library.members = old['members']
            portfolio.changed()
            policy.checkpoint(output / old['policy_file'], load=True)
            begin, totals = old['epoch'], old['totals']
            started -= old['elapsed_seconds']     # totals continue, so the clock they are divided by must too
            state['accepted'] = old['accepted']
            rebuild = Scorer(portfolio, archive, cfg.rho_max)
            rebuild.evaluate(old['archive'])
            archive.update(rebuild.pending)
            del rebuild
        state['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        initial = portfolio.reference
        state.update(initial_sharpe=initial.sharpe, initial_cagr=initial.cagr,
                     candidate_batch=engine.candidate_batch, trial_batch=engine.trial_batch)
        for epoch in range(begin, cfg.epochs):
            if (output / 'STOP').exists():
                state['phase'] = 'stopped'
                break
            evaluator = Scorer(portfolio, archive, cfg.rho_max)
            frozen_version = evaluator.version
            frozen_archive = evaluator.archive_version
            state['gate'] = evaluator.gate
            state.update(phase='sampling', epoch=epoch + 1)
            write_json(status_path, state)
            sample_start = time.perf_counter()
            batch = policy.sample(cfg.batch_size if args.mode == 'gfn' else cfg.batch_size + 12 * cfg.probes,
                                  cfg.probes if args.mode == 'gfn' else 0,
                                  evaluator.context(), args.mode == 'random')
            sampling_seconds = time.perf_counter() - sample_start
            state.update(phase='evaluating', generated=len(batch['expressions']))
            write_json(status_path, state)
            eval_start = time.perf_counter()
            reports = []
            # Bounded chunks also keep the independent dashboard responsive.
            for lo in range(0, len(batch['expressions']), engine.candidate_batch):
                reports.extend(evaluator.evaluate(batch['expressions'][lo:lo + engine.candidate_batch]))
                state.update({k: totals.get(k, 0) + v for k, v in evaluator.counters.items()})
                state['recent'] = reports[-8:]
                free, total = cp.cuda.runtime.memGetInfo()
                state['gpu_used_mib'] = round((total - free) / 1048576)
                state['elapsed_seconds'] = round(time.perf_counter() - started, 2)
                write_json(status_path, state)
            eval_seconds = time.perf_counter() - eval_start
            # The policy trains on CPU cores (C++, GIL released) while the GPU does admission and
            # reweighting; neither reads the other's result, and both finish before the next sample.
            state['phase'] = 'training'
            write_json(status_path, state)
            training = trainer.submit(train, policy, [r['log_reward'] for r in reports], batch['epoch'], cfg) if args.mode == 'gfn' else None
            candidate = evaluator.accept()
            pending = evaluator.pending
            counters = evaluator.counters
            del evaluator
            state['accepted'] += int(candidate is not None)
            # The reweight objective freezes the portfolio AFTER candidate admission.
            state['phase'] = 'reweighting'
            write_json(status_path, state)
            reweight = None
            reweight_start = time.perf_counter()
            if cfg.reweight_iterations and (candidate or epoch % cfg.reweight_every == 0):
                optim = Evaluator(portfolio, archive, cfg.rho_max)
                reweight = optim.reweight(cfg.reweight_iterations, cfg.reweight_batch, cfg.seed + epoch)
                del optim
            reweight_seconds = time.perf_counter() - reweight_start
            learned, train_seconds = training.result() if training else ({}, 0.)
            archive.update(pending)
            pending.clear()
            if args.accept and (candidate or (reweight and reweight['changed'])):
                library.save()
            for key, value in counters.items():
                totals[key] += value
            totals['probe_evaluations'] += batch['probe_count'] * 12
            totals['reweight_evaluations'] += reweight['evaluations'] if reweight else 0
            current = portfolio.reference
            state.update(totals, **learned, phase='checkpoint',
                         portfolio_sharpe=current.sharpe, portfolio_cagr=current.cagr,
                         sampling_seconds=sampling_seconds, evaluation_seconds=eval_seconds,
                         training_seconds=train_seconds,
                         reweight_seconds=reweight_seconds,
                         evaluation_candidates_per_second=round(counters['unique_evaluations'] / eval_seconds, 2),
                         candidates_per_second=round(totals['unique_evaluations'] / (time.perf_counter() - started), 2))
            record = dict(epoch=epoch + 1, portfolio_version=frozen_version, archive_version=frozen_archive,
                          signature=signature, candidate=candidate, reweight=reweight, training=learned,
                          sampling_seconds=sampling_seconds, evaluation_seconds=eval_seconds,
                          training_seconds=train_seconds, reports=reports, counters=counters)
            record['reweight_seconds'] = reweight_seconds
            record['gate'] = state['gate']
            with (output / 'epochs.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            policy_file = f'policy-{epoch + 1}.bin'
            policy.checkpoint(output / policy_file)
            write_json(output / 'state.json', dict(epoch=epoch + 1, signature=signature, members=library.members,
                       totals=totals, accepted=state['accepted'], policy_file=policy_file,
                       archive=[v['expr'] for v in archive.items.values()],
                       elapsed_seconds=time.perf_counter() - started))
            write_json(status_path, state)
            print(json.dumps({k: state[k] for k in ('epoch', 'unique_evaluations', 'g2', 'low_corr', 'accepted',
                              'portfolio_sharpe', 'portfolio_cagr', 'candidates_per_second')}, ensure_ascii=False), flush=True)
        else:
            state['phase'] = 'complete'
        state['elapsed_seconds'] = round(time.perf_counter() - started, 2)
        write_json(status_path, state)
    except BaseException as exc:
        state.update(phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        write_json(status_path, state)
        raise
    finally:
        trainer.shutdown()                  # a training step still running holds the native policy
        lock.unlink(missing_ok=True)

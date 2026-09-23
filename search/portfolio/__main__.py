"""python -m search.portfolio OUTPUT_DIR [--accept] [--mode random|gfn] [--seed-portfolio equal] [--null-shift]."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)
    if path.name == 'status.json':
        latest = Path(__file__).resolve().parents[2]/'tmp/q25_machine/live.json'
        latest.parent.mkdir(parents=True, exist_ok=True)
        live_temp = latest.with_suffix(f'.{os.getpid()}.tmp')
        live_temp.write_text(json.dumps(dict(value, run=str(path.parent)), ensure_ascii=False, allow_nan=False), encoding='utf-8')
        os.replace(live_temp, latest)


def panel(path):
    from monitoring.q25 import watch
    watch(path)


def train(policy, rewards, epoch, cfg):
    start = time.perf_counter()
    learned = policy.train(rewards, epoch, cfg.train_steps, cfg.learning_rate, cfg.opd_strength)
    return learned, time.perf_counter()-start


def run(args):
    from config.config import Q25SearchConfig
    from .native import Policy
    from .evaluator import Archive, Evaluator, engine, gates, combolib, cp
    cfg = Q25SearchConfig()
    overrides = {k: getattr(args, k) for k in ('epochs', 'batch_size', 'probes', 'seed') if getattr(args, k) is not None}
    cfg = replace(cfg, **overrides)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    if (output / 'state.json').exists() and not args.resume:
        raise ValueError('Output already has a run; use --resume or a new output directory')
    lock = output / '.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    started = time.perf_counter()
    # objective / null_shift tell the dashboard which funnel, verdicts and candidate columns this run has (monitoring/objectives.py)
    state = dict(phase='loading', epoch=0, epochs=cfg.epochs, mode=args.mode, objective=args.objective,
                 null_shift=args.null_shift, accepted=0)
    write_json(status_path, state)
    policy = None
    trainer = ThreadPoolExecutor(max_workers=1)
    try:
        data = engine.load()
        from config.config import FactorMiningConfig
        data_dir = Path(FactorMiningConfig().qnt_data_dir)
        if args.null_shift and args.seed_portfolio != 'equal':
            raise ValueError('The noise control delays every expression book, members included: start it from --seed-portfolio equal')
        engine.NULL_SHIFT = args.null_shift
        if args.objective == 'single':
            if args.accept or args.null_shift:
                raise ValueError('The single-alpha objective is an experiment: it cannot write the production portfolio, '
                                 'and its gate needs undelayed candidates to measure against their delayed twins')
            from . import single
            Scorer, objective = single.SingleEvaluator, asdict(single.CFG)
        else:
            Scorer, objective = Evaluator, None
        signature = dict(protocol=gates.PROTOCOL, config=asdict(cfg), combo=asdict(gates.CFG), mode=args.mode,
                         seed_portfolio=args.seed_portfolio, null_shift=args.null_shift,
                         objective=args.objective, single=objective,
                         code=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*'))
                                                     if p.suffix in ('.py','.cpp') and not p.name.startswith('test_'))).hexdigest(),
                         data=[(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(data_dir.glob('*.parquet'))],
                         vocabulary=hashlib.sha256('\n'.join(engine.operands()).encode()).hexdigest(),
                         precision=asdict(engine.CUDA))
        from gpu import MAX_W_1, MAX_W_2
        policy = Policy(engine.operands(), cfg.seed, cfg.max_tokens, cfg.max_cost, cfg.max_ts_depth, (MAX_W_1,MAX_W_2))
        lib = combolib.ComboLib.load()
        if args.seed_portfolio == 'equal':
            if args.accept:
                raise ValueError('An equal-weight experiment cannot overwrite the production portfolio')
            lib = combolib.ComboLib([])
        archive = Archive(cfg.archive_size)
        begin = 0
        totals = dict.fromkeys(Scorer.COUNTERS+('probe_evaluations', 'reweight_evaluations'), 0)
        if args.resume:
            old = json.loads((output/'state.json').read_text(encoding='utf-8'))
            # Compare JSON forms so tuple/list differences do not break signatures.
            if json.loads(json.dumps(signature)) != old['signature']:
                raise ValueError('Resume data/config/protocol mismatch')
            if args.accept and old['members'] != lib.members:
                raise ValueError('Production portfolio changed since checkpoint')
            lib.members = old['members']; lib._ref.clear()
            policy.checkpoint(output/old['policy_file'], load=True)
            begin, totals = old['epoch'], old['totals']
            started -= old['elapsed_seconds']     # totals continue, so the clock they are divided by must too
            state['accepted'] = old['accepted']
            rebuild = Scorer(lib, archive, cfg.rho_max)
            rebuild.evaluate(old['archive'])
            archive.update(rebuild.pending)
            del rebuild
        state['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        initial = lib.ref()
        state.update(initial_sharpe=initial['sharpe'], initial_cagr=initial['cagr'],
                     candidate_batch=engine.CUDA.candidate_batch, trial_batch=engine.CUDA.trial_batch)
        for epoch in range(begin, cfg.epochs):
            if (output/'STOP').exists():
                state['phase'] = 'stopped'; break
            evaluator = Scorer(lib, archive, cfg.rho_max)
            frozen_version = evaluator.version
            frozen_archive = evaluator.archive_version
            state['gate'] = evaluator.gate
            state.update(phase='sampling', epoch=epoch+1)
            write_json(status_path, state)
            sample_start = time.perf_counter()
            batch = policy.sample(cfg.batch_size if args.mode == 'gfn' else cfg.batch_size+12*cfg.probes,
                                  cfg.probes if args.mode == 'gfn' else 0,
                                  evaluator.context(), args.mode)
            sampling_seconds = time.perf_counter()-sample_start
            state.update(phase='evaluating', generated=len(batch['expressions']))
            write_json(status_path, state)
            eval_start = time.perf_counter()
            reports = []
            # Bounded chunks also keep the independent shell panel responsive.
            for lo in range(0, len(batch['expressions']), engine.CUDA.candidate_batch):
                reports.extend(evaluator.evaluate(batch['expressions'][lo:lo+engine.CUDA.candidate_batch]))
                state.update({k: totals.get(k, 0)+v for k,v in evaluator.counters.items()})
                state['recent'] = reports[-8:]
                free, total = cp.cuda.runtime.memGetInfo()
                state['gpu_used_mib'] = round((total-free)/1048576)
                state['elapsed_seconds'] = round(time.perf_counter()-started, 2)
                write_json(status_path, state)
            eval_seconds = time.perf_counter()-eval_start
            # The policy trains on CPU cores (C++, GIL released) while the GPU does admission and
            # reweighting; neither reads the other's result, and both finish before the next sample.
            state['phase'] = 'training'; write_json(status_path, state)
            training = trainer.submit(train, policy, [r['log_reward'] for r in reports], batch['epoch'], cfg) if args.mode == 'gfn' else None
            candidate = evaluator.accept()
            pending = evaluator.pending
            counters = evaluator.counters
            del evaluator
            state['accepted'] += int(candidate is not None)
            # The reweight objective freezes the portfolio AFTER candidate admission.
            state['phase'] = 'reweighting'; write_json(status_path, state)
            reweight = None
            reweight_start = time.perf_counter()
            if cfg.reweight_iterations and (candidate or epoch % cfg.reweight_every == 0):
                optim = Evaluator(lib, archive, cfg.rho_max)
                reweight = optim.reweight(cfg.reweight_iterations, cfg.reweight_batch, cfg.seed+epoch)
                del optim
            reweight_seconds = time.perf_counter()-reweight_start
            learned, train_seconds = training.result() if training else ({}, 0.)
            archive.update(pending)
            pending.clear()
            if args.accept and (candidate or (reweight and reweight['changed'])):
                lib.save()
            for key, value in counters.items():
                totals[key] += value
            totals['probe_evaluations'] += batch['probe_count']*12
            totals['reweight_evaluations'] += reweight['evaluations'] if reweight else 0
            current = lib.ref()
            state.update(totals, **learned, phase='checkpoint',
                         portfolio_sharpe=current['sharpe'], portfolio_cagr=current['cagr'],
                         sampling_seconds=sampling_seconds, evaluation_seconds=eval_seconds,
                         training_seconds=train_seconds,
                         reweight_seconds=reweight_seconds,
                         evaluation_candidates_per_second=round(counters['unique_evaluations']/eval_seconds,2),
                         candidates_per_second=round(totals['unique_evaluations']/(time.perf_counter()-started), 2))
            record = dict(epoch=epoch+1, portfolio_version=frozen_version, archive_version=frozen_archive,
                          signature=signature, candidate=candidate, reweight=reweight, training=learned,
                          sampling_seconds=sampling_seconds, evaluation_seconds=eval_seconds,
                          training_seconds=train_seconds, reports=reports, counters=counters)
            record['reweight_seconds'] = reweight_seconds
            record['gate'] = state['gate']
            with (output/'epochs.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            policy_file = f'policy-{epoch+1}.bin'
            policy.checkpoint(output/policy_file)
            write_json(output/'state.json', dict(epoch=epoch+1, signature=signature, members=lib.members,
                       totals=totals, accepted=state['accepted'], policy_file=policy_file,
                       archive=[v['expr'] for v in archive.items.values()],
                       elapsed_seconds=time.perf_counter()-started))
            write_json(status_path, state)
            print(json.dumps({k:state[k] for k in ('epoch','unique_evaluations','g2','low_corr','accepted',
                              'portfolio_sharpe','portfolio_cagr','candidates_per_second')}, ensure_ascii=False), flush=True)
        else:
            state['phase'] = 'complete'
        state['elapsed_seconds'] = round(time.perf_counter()-started, 2)
        write_json(status_path, state)
    except BaseException as exc:
        state.update(phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        write_json(status_path, state)
        raise
    finally:
        trainer.shutdown()                  # a step still running holds the native handle
        if policy:
            policy.close()
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description='C++ portfolio-conditioned GFlowNet + CUDA Q25 evaluation')
    p.add_argument('output', nargs='?', help='Run directory (checkpoints, JSONL and live status)')
    p.add_argument('--watch', metavar='STATUS_JSON', help='Read-only independent shell dashboard')
    p.add_argument('--accept', action='store_true', help='Persist verified changes to the production portfolio')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--mode', choices=('gfn','random'), default='gfn')
    p.add_argument('--seed-portfolio', choices=('current','equal'), default='current')
    p.add_argument('--objective', choices=('portfolio','single'), default='portfolio',
                   help='portfolio: reward = what the candidate adds to the portfolio; single: reward = the candidate alone, '
                        'admission by the (correlation, Sharpe) gate of search/portfolio/single.py')
    p.add_argument('--null-shift', action='store_true',
                   help='Noise control: every candidate signal is delayed by its own fixed 1-3 years before it becomes a book')
    for name in ('epochs','batch-size','probes','seed'):
        p.add_argument('--'+name, type=int)
    args = p.parse_args()
    if args.watch:
        panel(args.watch)
    elif args.output:
        run(args)
    else:
        p.error('Specify an output directory or --watch')


if __name__ == '__main__':
    main()

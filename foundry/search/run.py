"""Search job lifecycle, shared research calls and checkpoint publication."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from foundry.adapters.files import publish
from foundry.adapters.library import FileLibraryStore, MemoryLibraryStore, member_from_json
from foundry.adapters.arrays import FileArrayStore
from foundry.adapters.serialization import json_value
from foundry.config import ROOT, Settings, path
from foundry.portfolio.library import PROTOCOL, Library
from foundry.research import Research
from foundry.runtime import configured_study, open_runtime
from foundry.search.evaluator import Archive, Evaluator
from foundry.search.policy import Policy
from foundry.search.single import SingleEvaluator

LIVE = ROOT / 'tmp' / 'monitor' / 'live.json'


def write_json(path, value):
    value = json_value(value)
    publish(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
    if path.name == 'status.json':
        LIVE.parent.mkdir(parents=True, exist_ok=True)
        publish(LIVE, json.dumps(dict(value, run=str(path.parent)), ensure_ascii=False, allow_nan=False).encode('utf-8'))


def train(policy, rewards, epoch, cfg):
    start = time.perf_counter()
    learned = policy.train(rewards, epoch, cfg.train_steps, cfg.learning_rate, cfg.opd_strength)
    return learned, time.perf_counter() - start


def run(output: Path, settings: Settings):
    run_started = time.perf_counter()
    cfg = settings.search
    if cfg.null_shift and cfg.seed_portfolio != 'equal':
        raise ValueError('The noise control delays every expression book: set search.seed_portfolio=equal')
    if cfg.seed_portfolio == 'equal' and cfg.accept:
        raise ValueError('An equal-weight experiment cannot overwrite the production portfolio')
    if cfg.objective == 'single' and (cfg.accept or cfg.null_shift):
        raise ValueError('The single-alpha experiment requires an in-memory portfolio and undelayed candidates')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    lock = output / '.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    state = dict(phase='loading', market=settings.market.name, epoch=0, epochs=cfg.epochs,
                 mode=cfg.mode, objective=cfg.objective, null_shift=cfg.null_shift, accepted=0,
                 checkpoint_seconds_total=0.)
    try:
        write_json(status_path, state)
        runtime_started = time.perf_counter()
        with open_runtime(settings, configured_study(settings)) as runtime, ThreadPoolExecutor(max_workers=1) as trainer:
            state['runtime_open_seconds'] = time.perf_counter() - runtime_started
            with runtime.resources.scope():
                _epochs(output, settings, runtime, trainer, state)
        state['run_wall_seconds'] = time.perf_counter() - run_started
        write_json(status_path, state)
    except BaseException as exc:
        state.update(phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        write_json(status_path, state)
        raise
    finally:
        lock.unlink()


def _epochs(output, settings, runtime, trainer, state):
    cfg, engine = settings.search, runtime.engine
    status_path = output / 'status.json'
    started = time.perf_counter()
    Scorer = SingleEvaluator if cfg.objective == 'single' else Evaluator
    signature = dict(snapshot=asdict(engine.context), protocol=PROTOCOL, config=asdict(settings),
                     candidate_batch=engine.candidate_batch, trial_batch=engine.trial_batch)
    policy = Policy(engine.vocabulary, cfg.seed, cfg.max_tokens, settings.expr.max_cost, cfg.max_ts_depth,
                    (engine.backend.max_window, engine.backend.max_pair_window))
    disk_store = FileLibraryStore(ROOT / 'production' / 'library.json')
    library = Library((), engine.study) if cfg.seed_portfolio == 'equal' else disk_store.read(engine.study)
    archive = Archive(cfg.archive_size)
    begin = 0
    totals = dict.fromkeys(Scorer.COUNTERS + ('probe_evaluations', 'reweight_evaluations'), 0)
    resume = (output / 'state.json').exists()
    if resume:
        old = json.loads((output / 'state.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(signature)) != old['signature']:
            raise ValueError('Resume data/config/protocol mismatch')
        if cfg.accept and old['members'] != [asdict(member) for member in library.members]:
            raise ValueError('Production portfolio changed since checkpoint')
        library = Library(tuple(member_from_json(member) for member in old['members']), engine.study)
        policy.checkpoint(output / old['policy_file'], load=True)
        begin, totals = old['epoch'], old['totals']
        started -= old['elapsed_seconds']
        state['accepted'] = old['accepted']
    store = disk_store if cfg.accept else MemoryLibraryStore(library)
    research = Research(engine, store, FileArrayStore(path(settings.portfolio.artifact_dir)), null_shift=cfg.null_shift)
    if resume:
        rebuild = Scorer(research, archive, cfg.rho_max)
        rebuild.evaluate(old['archive'])
        archive.update(rebuild.pending)
        del rebuild
    resource_status = runtime.status()
    initial = research.portfolio.reference
    state.update(device=resource_status.device, initial_sharpe=initial.sharpe, initial_cagr=initial.cagr,
                 initial_maxdd=initial.maxdd, initial_wsharpe=initial.wsharpe,
                 candidate_batch=engine.candidate_batch, trial_batch=engine.trial_batch)
    for epoch in range(begin, cfg.epochs):
        if (output / 'STOP').exists():
            state['phase'] = 'stopped'
            break
        evaluator = Scorer(research, archive, cfg.rho_max)
        frozen_version, frozen_archive = evaluator.version, evaluator.archive_version
        state['gate'] = asdict(evaluator.gate) if evaluator.gate is not None else None
        state.update(phase='sampling', epoch=epoch + 1)
        write_json(status_path, state)
        sample_start = time.perf_counter()
        batch = policy.sample(cfg.batch_size if cfg.mode == 'gfn' else cfg.batch_size + 12 * cfg.probes,
                              cfg.probes if cfg.mode == 'gfn' else 0, evaluator.context(), cfg.mode == 'random')
        sampling_seconds = time.perf_counter() - sample_start
        state.update(phase='evaluating', generated=len(batch['expressions']))
        write_json(status_path, state)
        eval_start = time.perf_counter()
        reports = []
        for lo in range(0, len(batch['expressions']), engine.candidate_batch):
            reports.extend(evaluator.evaluate(batch['expressions'][lo:lo + engine.candidate_batch]))
            state.update({key: totals[key] + value for key, value in evaluator.counters.items()})
            state['recent'] = [asdict(row) for row in reports[-8:]]
            state['gpu_used_mib'] = runtime.status().used_mib
            state['elapsed_seconds'] = round(time.perf_counter() - started, 2)
            write_json(status_path, state)
        eval_seconds = time.perf_counter() - eval_start
        state['phase'] = 'training'
        write_json(status_path, state)
        training = trainer.submit(train, policy, [row.log_reward for row in reports], batch['epoch'], cfg) if cfg.mode == 'gfn' else None
        commit_start = time.perf_counter()
        candidate = evaluator.accept()
        commit_seconds = time.perf_counter() - commit_start
        pending, counters = evaluator.pending, evaluator.counters
        del evaluator
        state['accepted'] += int(candidate is not None)
        state['phase'] = 'reweighting'
        write_json(status_path, state)
        reweight = None
        reweight_start = time.perf_counter()
        if cfg.reweight_iterations and (candidate is not None or epoch % cfg.reweight_every == 0):
            optim = Evaluator(research, archive, cfg.rho_max)
            reweight = optim.reweight(cfg.reweight_iterations, cfg.reweight_batch, cfg.seed + epoch)
            del optim
        reweight_seconds = time.perf_counter() - reweight_start
        learned, train_seconds = training.result() if training is not None else ({}, 0.)
        archive.update(pending)
        pending.clear()
        for key, value in counters.items():
            totals[key] += value
        totals['probe_evaluations'] += batch['probe_count'] * 12
        totals['reweight_evaluations'] += reweight.evaluations if reweight is not None else 0
        current = research.portfolio.reference
        state.update(totals, **learned, phase='checkpoint', portfolio_sharpe=current.sharpe,
                     portfolio_cagr=current.cagr, portfolio_maxdd=current.maxdd, portfolio_wsharpe=current.wsharpe,
                     sampling_seconds=sampling_seconds, commit_seconds=commit_seconds,
                     evaluation_seconds=eval_seconds, training_seconds=train_seconds,
                     reweight_seconds=reweight_seconds,
                     evaluation_candidates_per_second=round(counters['unique_evaluations'] / eval_seconds, 2),
                     candidates_per_second=round(totals['unique_evaluations'] / (time.perf_counter() - started), 2))
        record = dict(epoch=epoch + 1, portfolio_version=frozen_version, archive_version=frozen_archive,
                      signature=signature, candidate=asdict(candidate) if candidate is not None else None,
                      reweight=asdict(reweight) if reweight is not None else None, training=learned,
                      sampling_seconds=sampling_seconds, evaluation_seconds=eval_seconds,
                      training_seconds=train_seconds, commit_seconds=commit_seconds, reweight_seconds=reweight_seconds,
                      reports=[asdict(row) for row in reports], counters=counters, gate=state['gate'])
        checkpoint_start = time.perf_counter()
        with (output / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(json_value(record), ensure_ascii=False, allow_nan=False) + '\n')
        policy_file = f'policy-{epoch + 1}.bin'
        policy.checkpoint(output / policy_file)
        write_json(output / 'state.json', dict(epoch=epoch + 1, signature=signature,
                   members=[asdict(member) for member in research.portfolio.library.members], totals=totals,
                   accepted=state['accepted'], policy_file=policy_file,
                   archive=[value.expr for value in archive.items.values()],
                   elapsed_seconds=time.perf_counter() - started))
        write_json(status_path, state)
        state['checkpoint_seconds_total'] += time.perf_counter() - checkpoint_start
        print(json.dumps({key: state[key] for key in ('epoch', 'unique_evaluations', 'g2', 'low_corr', 'accepted',
                          'portfolio_sharpe', 'portfolio_cagr', 'candidates_per_second')}, ensure_ascii=False), flush=True)
    else:
        state['phase'] = 'complete'
    state['elapsed_seconds'] = round(time.perf_counter() - started, 2)
    write_json(status_path, state)

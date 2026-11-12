"""python -m foundry <command> — the one entry point.

  formulas     eval [FORMULA ...] [--file F] [--accept] [--json OUT] [--quiet] [--in-process]
               family FILE [--json OUT] [--in-process]      (a `#` line is the hypothesis of the formulas below it)
               fields | ops                                 (what formulas can read / use)
  service      serve | status | stop | attach [HOST] | detach | mirror HOST REMOTE_STATUS LOCAL_STATUS
  search       mine OUT [--accept] [--resume] [--mode gfn|random] [--seed-portfolio current|equal]
                    [--objective portfolio|single] [--null-shift] [--epochs N] [--batch-size N] [--probes N] [--seed N]
               calibrate RUN_DIR                            (reward violation scales from a finished run)
  dashboard    monitor [--source STATUS] [--attach] [--once] | preview [STATUS ...]

Evaluation goes through the resident service (started on demand, or the attached cloud one); --in-process computes
here instead. Only --accept writes the production library.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _formulas(a):
    exprs = list(a.formulas)
    if a.file:
        exprs += [l.strip() for l in Path(a.file).read_text(encoding='utf-8').splitlines() if l.strip() and not l.strip().startswith('#')]
    if not exprs:
        sys.exit('no formulas: give formulas or --file')
    return exprs


def _in_process_portfolio():
    from foundry.engine import Engine
    from foundry.portfolio.library import Library
    from foundry.portfolio.portfolio import Portfolio
    return Portfolio(Engine(), Library.load())


def cmd_eval(a):
    from foundry import report
    exprs = _formulas(a)
    if a.in_process:
        from foundry.service.daemon import evaluate
        rows = evaluate(_in_process_portfolio(), exprs, a.accept)
    else:
        from foundry.service import client
        rows = client.evaluate(exprs, accept=a.accept, quiet=a.quiet, source=a.file and Path(a.file).as_posix())
    report.rows(rows, a.quiet)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


def cmd_family(a):
    from foundry import report
    seeds, hypothesis = [], ''
    for line in Path(a.file).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.startswith('#'):
            hypothesis = line.lstrip('#').strip()
        elif line:
            seeds.append(dict(expr=line, hypothesis=hypothesis))
    if not seeds:
        sys.exit(f'{a.file}: no formulas')
    if a.in_process:
        from foundry.portfolio import family
        res = family.expand(_in_process_portfolio(), seeds)
    else:
        from foundry.service import client
        res = client.evaluate_family(seeds, Path(a.file).as_posix())
    report.family(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


def cmd_fields(a):
    from foundry import report
    report.fields()


def cmd_ops(a):
    from foundry import report
    report.ops()


def cmd_serve(a):
    from foundry.service import client
    from foundry.service.daemon import Server
    d = client.info()
    if not (d and d.get('host')):          # a local service would hold a second copy of the panel on the card
        client.stop()
    client.PORTFILE.unlink(missing_ok=True)
    Server(0).run()


def cmd_status(a):
    from foundry.service import client
    p = client.request(dict(op='ping'), timeout=10.0)
    print(json.dumps(p, ensure_ascii=False, indent=2) if p else '未运行')


def cmd_stop(a):
    from foundry.service import client
    print(client.stop() or '未运行')
    client.PORTFILE.unlink(missing_ok=True)


def cmd_attach(a):
    from foundry.service import client, remote
    d = client.info()
    if d and not d.get('host'):            # free the local card before attaching to the cloud
        print(client.stop() or '(无本地服务)')
    remote.attach(a.host or remote.DEFAULT_HOST)


def cmd_detach(a):
    from foundry.service import client
    client.PORTFILE.unlink(missing_ok=True)
    print('已断开;云端服务未动(要停用先 attach 再 stop)')


def cmd_mirror(a):
    from foundry.config import MonitorConfig
    from foundry.service import remote
    remote.mirror(a.host, a.remote, Path(a.local), MonitorConfig().push_seconds)


def cmd_mine(a):
    from foundry.search import run
    run.run(a)


def cmd_calibrate(a):
    from foundry.search import calibrate
    calibrate.report(Path(a.run))


def cmd_monitor(a):
    from foundry.monitor import launcher
    launcher.main(a.source, a.attach, a.once)


def cmd_preview(a):
    from foundry.monitor import preview
    print(preview.write(a.statuses))


def main():
    p = argparse.ArgumentParser(prog='python -m foundry', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('eval')
    s.add_argument('formulas', nargs='*')
    s.add_argument('--file')
    s.add_argument('--accept', action='store_true', help='admit the passing formulas into the production library')
    s.add_argument('--json')
    s.add_argument('--quiet', action='store_true', help='print only admitted formulas and a count per stage')
    s.add_argument('--in-process', action='store_true', help='compute here instead of in the service')
    s = sub.add_parser('family')
    s.add_argument('file')
    s.add_argument('--json')
    s.add_argument('--in-process', action='store_true')
    sub.add_parser('fields')
    sub.add_parser('ops')
    sub.add_parser('serve')
    sub.add_parser('status')
    sub.add_parser('stop')
    s = sub.add_parser('attach')
    s.add_argument('host', nargs='?')
    sub.add_parser('detach')
    s = sub.add_parser('mirror')
    s.add_argument('host')
    s.add_argument('remote')
    s.add_argument('local')
    s = sub.add_parser('mine')
    s.add_argument('output')
    s.add_argument('--accept', action='store_true', help='persist verified changes to the production library')
    s.add_argument('--resume', action='store_true')
    s.add_argument('--mode', choices=('gfn', 'random'), default='gfn')
    s.add_argument('--seed-portfolio', choices=('current', 'equal'), default='current')
    s.add_argument('--objective', choices=('portfolio', 'single'), default='portfolio')
    s.add_argument('--null-shift', action='store_true', help='noise control: every signal delayed by its own 1-3 years')
    for name in ('epochs', 'batch-size', 'probes', 'seed'):
        s.add_argument('--' + name, type=int)
    s = sub.add_parser('calibrate')
    s.add_argument('run')
    s = sub.add_parser('monitor')
    s.add_argument('--source', type=Path)
    s.add_argument('--attach', action='store_true', help='render in this terminal')
    s.add_argument('--once', action='store_true', help='print one frame and exit')
    s = sub.add_parser('preview')
    s.add_argument('statuses', nargs='*', type=Path)
    a = p.parse_args()
    globals()['cmd_' + a.command.replace('-', '_')](a)


if __name__ == '__main__':
    main()

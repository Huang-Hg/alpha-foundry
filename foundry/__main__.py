"""python -m foundry <command> — explicit service, JSON-RPC requests, and mining.

  schema OUTPUT                     write the machine-readable request contract
  request INPUT.jsonl OUTPUT.jsonl   send JSON-RPC requests to the configured service
  fields | ops                      inspect the active market vocabulary
  serve | status | stop | attach [HOST] | detach | mirror HOST REMOTE LOCAL
  mine OUTPUT                       run the configured mining program
  import-perpetual OUTPUT            import verified public USD-M archives
  benchmark OUTPUT                  validate and compare native engine performance
  calibrate RUN | monitor | preview

Start or attach the service explicitly. request never starts a local GPU process.
"""
from __future__ import annotations

from foundry.config import configure, parameters_from_json, ROOT


import argparse
import json
import sys
from pathlib import Path


def cmd_request(a):
    from foundry.rpc.client import invoke
    with open(a.output,'w',encoding='utf-8') as out:
        for line in Path(a.input).read_text(encoding='utf-8').splitlines():
            response=invoke(json.loads(line))
            out.write(json.dumps(response,ensure_ascii=False,allow_nan=False)+'\n')
            out.flush()


def cmd_schema(a):
    from foundry.rpc.schema import describe
    Path(a.output).write_text(json.dumps(describe(),ensure_ascii=False,indent=2),encoding='utf-8')


def cmd_fields(a):
    from foundry.study import report
    from foundry.study.resources import open_runtime, configured_study
    settings = configure(**a.parameters)
    with open_runtime(settings, configured_study(settings)) as runtime:
        report.fields(runtime)


def cmd_ops(a):
    from foundry.study import report
    report.ops(configure(**a.parameters).expr)


def cmd_serve(a):
    from foundry.rpc import client, schema
    from foundry import serve
    if schema.PORTFILE.exists():
        client.stop()
    serve(ROOT / 'production' / 'library.json', **a.parameters)


def cmd_status(a):
    from foundry.rpc import client, schema
    p = client.call('system.status',{},timeout=10.)
    print(json.dumps(p, ensure_ascii=False, indent=2) if p else '未运行')


def cmd_stop(a):
    from foundry.rpc import client, schema
    print(client.stop() or '未运行')
    schema.PORTFILE.unlink(missing_ok=True)


def cmd_attach(a):
    from foundry.rpc import client, schema
    d = client.info() if schema.PORTFILE.exists() else {}
    if d and 'host' not in d:            # free the local card before attaching to the cloud
        print(client.stop() or '(无本地服务)')
    client.attach(a.host or client.DEFAULT_HOST)


def cmd_detach(a):
    from foundry.rpc import client, schema
    schema.PORTFILE.unlink(missing_ok=True)
    print('已断开;云端服务未动(要停用先 attach 再 stop)')


def cmd_mirror(a):
    from foundry.rpc import client, schema
    client.mirror(a.host, a.remote, Path(a.local), configure(**a.parameters).monitor.push_seconds)


def cmd_mine(a):
    from foundry import mine
    mine(Path(a.output), **a.parameters)


def cmd_calibrate(a):
    from foundry.search import calibrate
    settings = configure(**a.parameters)
    calibrate.report(Path(a.run), settings.search)


def cmd_monitor(a):
    from foundry.monitor import launcher
    from foundry.study.store import publish
    launcher.main(a.source, a.attach, a.once, configure(**a.parameters).monitor, publish=publish)


def cmd_preview(a):
    from foundry.monitor import preview
    print(preview.write(a.statuses))


def cmd_import_perpetual(a):
    from foundry import import_perpetual
    manifest = import_perpetual(Path(a.output), **a.parameters)
    print(json.dumps(dict(output=a.output, sources=len(manifest['sources']),
                          artifacts=manifest['artifacts']), indent=2))


def cmd_benchmark(a):
    from foundry.api import benchmark
    print(benchmark(a.output, suite=a.suite, sizes=a.sizes, assets=a.assets,
                    spectre_python=a.spectre_python, qualification=a.qualification, backends=a.backends, **a.parameters))


def main():
    p = argparse.ArgumentParser(prog='python -m foundry', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parameters', default='{}', help='JSON object of public research parameters; see schema')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('request')
    s.add_argument('input')
    s.add_argument('output')
    s = sub.add_parser('schema')
    s.add_argument('output')
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
    s = sub.add_parser('import-perpetual')
    s.add_argument('output')
    s = sub.add_parser('benchmark')
    s.add_argument('output')
    s.add_argument('--suite', choices=('foundry', 'frameworks', 'discovery', 'nonformula', 'timed'), default='foundry')
    s.add_argument('--sizes', type=int, nargs='+')
    s.add_argument('--backends', nargs='+', choices=('cpu', 'cuda'), default=['cpu', 'cuda'])
    s.add_argument('--assets', type=int, default=8)
    s.add_argument('--spectre-python', help='isolated interpreter containing the pinned Spectre revision')
    s.add_argument('--qualification', type=Path, help='timed precision qualification evidence from the completed smoke')
    s = sub.add_parser('calibrate')
    s.add_argument('run')
    s = sub.add_parser('monitor')
    s.add_argument('--source', type=Path)
    s.add_argument('--attach', action='store_true', help='render in this terminal')
    s.add_argument('--once', action='store_true', help='print one frame and exit')
    s = sub.add_parser('preview')
    s.add_argument('statuses', nargs='*', type=Path)
    a = p.parse_args()
    from jsonschema import validate
    from foundry.rpc.schema import describe
    parameters = json.loads(a.parameters)
    validate(parameters, describe()['parameters'])
    if parameters and a.command not in {'fields', 'ops', 'serve', 'mine', 'import-perpetual', 'benchmark', 'calibrate', 'monitor', 'mirror'}:
        raise ValueError('parameters apply at compute/job startup, not when connecting to an existing service')
    a.parameters = parameters_from_json(parameters)
    globals()['cmd_' + a.command.replace('-', '_')](a)


if __name__ == '__main__':
    main()

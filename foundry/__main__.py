"""python -m foundry <command> — explicit service, JSON-RPC requests, and mining.

  schema OUTPUT                     write the machine-readable request contract
  request INPUT.jsonl OUTPUT.jsonl   send JSON-RPC requests to the configured service
  fields | ops                      inspect the active market vocabulary
  serve | status | stop | attach [HOST] | detach | mirror HOST REMOTE LOCAL
  mine OUTPUT                       run the configured mining program
  import-perpetual OUTPUT            import verified public USD-M archives
  calibrate RUN | monitor | preview

Start or attach the service explicitly. request never starts a local GPU process.
"""
from __future__ import annotations

from foundry.config import load_config, ROOT, path


import argparse
import json
import sys
from pathlib import Path


def cmd_request(a):
    from foundry.service.client import invoke
    with open(a.output,'w',encoding='utf-8') as out:
        for line in Path(a.input).read_text(encoding='utf-8').splitlines():
            response=invoke(json.loads(line))
            out.write(json.dumps(response,ensure_ascii=False,allow_nan=False)+'\n')
            out.flush()


def cmd_schema(a):
    from foundry.service.schema import describe
    Path(a.output).write_text(json.dumps(describe(),ensure_ascii=False,indent=2),encoding='utf-8')


def cmd_fields(a):
    from foundry import report
    from foundry.runtime import open_runtime, configured_study
    settings = load_config()
    with open_runtime(settings, configured_study(settings)) as runtime:
        report.fields(runtime)


def cmd_ops(a):
    from foundry import report
    report.ops(load_config().expr)


def cmd_serve(a):
    from foundry.service import client
    from foundry.service.daemon import Server
    if client.PORTFILE.exists():
        client.stop()
    from foundry.runtime import open_runtime, configured_study
    from foundry.adapters.library import FileLibraryStore
    from foundry.adapters.arrays import FileArrayStore
    from foundry.research import Research
    settings = load_config()
    with open_runtime(settings, configured_study(settings)) as runtime:
        store = FileLibraryStore(ROOT / 'production' / 'library.json')
        research = Research(runtime.engine, store, FileArrayStore(path(settings.portfolio.artifact_dir)))
        Server(runtime, research).run()


def cmd_status(a):
    from foundry.service import client
    p = client.call('system.status',{},timeout=10.)
    print(json.dumps(p, ensure_ascii=False, indent=2) if p else '未运行')


def cmd_stop(a):
    from foundry.service import client
    print(client.stop() or '未运行')
    client.PORTFILE.unlink(missing_ok=True)


def cmd_attach(a):
    from foundry.service import client, remote
    d = client.info() if client.PORTFILE.exists() else {}
    if d and 'host' not in d:            # free the local card before attaching to the cloud
        print(client.stop() or '(无本地服务)')
    remote.attach(a.host or remote.DEFAULT_HOST)


def cmd_detach(a):
    from foundry.service import client
    client.PORTFILE.unlink(missing_ok=True)
    print('已断开;云端服务未动(要停用先 attach 再 stop)')


def cmd_mirror(a):
    from foundry.service import remote
    remote.mirror(a.host, a.remote, Path(a.local), load_config().monitor.push_seconds)


def cmd_mine(a):
    from foundry.search import run
    run.run(Path(a.output), load_config())


def cmd_calibrate(a):
    from foundry.search import calibrate
    calibrate.report(Path(a.run), load_config().portfolio)


def cmd_monitor(a):
    from foundry.monitor import launcher
    launcher.main(a.source, a.attach, a.once, load_config().monitor)


def cmd_preview(a):
    from foundry.monitor import preview
    print(preview.write(a.statuses))


def cmd_import_perpetual(a):
    from foundry.adapters.perpetual_data import import_perpetual
    manifest = import_perpetual(Path(a.output), load_config().crypto_perpetual)
    print(json.dumps(dict(output=a.output, sources=len(manifest['sources']),
                          artifacts=manifest['artifacts']), indent=2))


def main():
    p = argparse.ArgumentParser(prog='python -m foundry', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

"""python -m monitoring [--source STATUS_JSON] [--attach] [--once].

Layout lives in view.py (blocks: components.py, per-objective content: objectives.py, drawing: cells.py);
`python -m monitoring.preview` exports the same frames to HTML.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'tmp/q25_monitor'
SOURCE = STATE/'source.json'
OWNER = STATE/'owner.json'


def write(path, value):
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)


def alive(owner):
    # The renderer acknowledges only after a frame has been flushed. A bounded
    # heartbeat works on Windows too, where os.kill(pid, 0) is not a POSIX probe.
    return owner.get('pid',0)>0 and 0<=time.time()-owner.get('heartbeat',0)<3


def main():
    parser=argparse.ArgumentParser(description='Q25 project dashboard (persistent named terminal)')
    parser.add_argument('--source',type=Path,help='Local status JSON; can be a mirrored remote status')
    parser.add_argument('--attach',action='store_true',help='Render in the current terminal')
    parser.add_argument('--once',action='store_true',help='Print one complete frame and exit')
    args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    if args.source or not SOURCE.exists():
        write(SOURCE,{'status_path':str((args.source or ROOT/'tmp/q25_machine/live.json').resolve())})
    if args.attach or args.once or os.name!='nt':
        from .q25 import watch
        watch(SOURCE,once=args.once,heartbeat=lambda:write(OWNER,{'pid':os.getpid(),'heartbeat':time.time()}))
        return
    try:
        owner=json.loads(OWNER.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):
        owner={}
    if alive(owner):
        print(f'Q25 dashboard is running (PID {owner["pid"]}); source updated.')
        return
    terminal=shutil.which('wt.exe')
    shell=shutil.which('pwsh.exe') or shutil.which('powershell.exe')
    if not terminal or not shell:
        raise RuntimeError('Windows Terminal/PowerShell required; use --attach in your existing shell')
    # Only this child environment is normalized. User/system settings are untouched.
    env=dict(os.environ,TERM='xterm-256color',COLORTERM='truecolor',PYTHONUTF8='1',PYTHONUNBUFFERED='1')
    env.pop('NO_COLOR',None)
    command=f"& '{sys.executable.replace(chr(39),chr(39)*2)}' -u -m monitoring --attach"
    startup=subprocess.STARTUPINFO()
    startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=1  # This command explicitly opens the user's dashboard.
    from .cells import BASE
    subprocess.Popen([terminal,'-w','q25-dashboard','new-tab','--title','Q25 Alpha Search','--tabColor',BASE,
                      '-d',str(ROOT),shell,'-NoProfile','-NoExit','-Command',command],
                     env=env,cwd=ROOT,startupinfo=startup)
    # Startup acknowledgement, not a blind 'window opened' success claim.
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        try:
            if alive(json.loads(OWNER.read_text(encoding='utf-8'))):
                print('Q25 full dashboard renderer started.');return
        except (FileNotFoundError,json.JSONDecodeError):
            pass
        time.sleep(.2)
    raise RuntimeError('Dashboard did not acknowledge startup; run python -m monitoring --attach for diagnostics')


if __name__=='__main__':
    main()

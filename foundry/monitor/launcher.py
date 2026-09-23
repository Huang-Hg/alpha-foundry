"""python -m foundry monitor [--source STATUS] [--attach] [--once]: the dashboard in its own named terminal window.

The renderer (watch.py) reads tmp/monitor/source.json, which names the status file to show, so a second call only
retargets the running window; owner.json is the renderer's heartbeat. `python -m foundry preview` exports the
same frames to HTML.
"""
import json
import os
import shutil
import subprocess
import sys
import time

from foundry.config import ROOT
from foundry.core.files import publish
from .cells import BASE
from .watch import watch

STATE = ROOT/'tmp'/'monitor'
SOURCE = STATE/'source.json'
OWNER = STATE/'owner.json'
LIVE = STATE/'live.json'                     # the search run mirrors its status here


def write(path, value):
    publish(path, json.dumps(value, ensure_ascii=False).encode('utf-8'))


def alive(owner):
    # The renderer acknowledges only after a frame has been flushed. A bounded
    # heartbeat works on Windows too, where os.kill(pid, 0) is not a POSIX probe.
    return owner.get('pid',0)>0 and 0<=time.time()-owner.get('heartbeat',0)<3


def main(source, attach, once):
    """source: status file to show from now on (None keeps the current one, the live search run at first);
    attach: render in this terminal; once: print one complete frame and exit."""
    STATE.mkdir(parents=True,exist_ok=True)
    if source or not SOURCE.exists():
        write(SOURCE,{'status_path':str((source or LIVE).resolve())})
    if attach or once or os.name!='nt':
        watch(SOURCE,once=once,heartbeat=lambda:write(OWNER,{'pid':os.getpid(),'heartbeat':time.time()}))
        return
    try:
        owner=json.loads(OWNER.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):
        owner={}
    if alive(owner):
        print(f'Dashboard is running (PID {owner["pid"]}); source updated.')
        return
    terminal=shutil.which('wt.exe')
    shell=shutil.which('pwsh.exe') or shutil.which('powershell.exe')
    if not terminal or not shell:
        raise RuntimeError('Windows Terminal/PowerShell required; use --attach in your existing shell')
    # Only this child environment is normalized. User/system settings are untouched.
    env=dict(os.environ,TERM='xterm-256color',COLORTERM='truecolor',PYTHONUTF8='1',PYTHONUNBUFFERED='1')
    env.pop('NO_COLOR',None)
    command=f"& '{sys.executable.replace(chr(39),chr(39)*2)}' -u -m foundry monitor --attach"
    startup=subprocess.STARTUPINFO()
    startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=1  # This command explicitly opens the user's dashboard.
    subprocess.Popen([terminal,'-w','foundry-dashboard','new-tab','--title','Alpha Search','--tabColor',BASE,
                      '-d',str(ROOT),shell,'-NoProfile','-NoExit','-Command',command],
                     env=env,cwd=ROOT,startupinfo=startup)
    # Startup acknowledgement, not a blind 'window opened' success claim.
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        try:
            if alive(json.loads(OWNER.read_text(encoding='utf-8'))):
                print('Dashboard renderer started.');return
        except (FileNotFoundError,json.JSONDecodeError):
            pass
        time.sleep(.2)
    raise RuntimeError('Dashboard did not acknowledge startup; run python -m foundry monitor --attach for diagnostics')

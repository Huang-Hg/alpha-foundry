"""One persistent terminal screen with switchable status sources.

Data rate and frame rate are decoupled: snapshots arrive every couple of seconds (a local producer, or
`python -m foundry mirror` for a cloud one), the screen is redrawn every [monitor] frame_seconds from
the local file only. Between snapshots the readings glide from the value on screen to the
newest reported one, so motion is smooth but nothing beyond a reported value is ever shown.
Which readings glide is declared in schema.py.
"""
import json
import os
from pathlib import Path
import sys
import time
from rich.console import Console
from rich.live import Live

from foundry.config import MonitorConfig
from .cells import BASE
from .schema import ENDED, GLIDE, RESTART
from .view import render


class Glide:
    """Linear motion from what is on screen to the latest snapshot, over one snapshot gap."""
    def __init__(self):
        self.target, self.start, self.arrived, self.span = {}, {}, 0., 1.

    def feed(self, state, now):
        shown = self.at(now)
        self.span = min(5., max(.2, now-self.arrived)) if self.target else 1.
        # A counter that moved backwards is a new run (or a restarted service): show it at once instead of counting down.
        restart = any(state.get(k) is not None and shown.get(k) is not None and state[k] < shown[k] for k in RESTART)
        self.start = {k: state[k] if restart or shown.get(k) is None else shown[k] for k in GLIDE if state.get(k) is not None}
        self.target, self.arrived = state, now

    def at(self, now):
        share = min(1., (now-self.arrived)/self.span)
        out = dict(self.target)
        for k, a in self.start.items():
            value = a+(self.target[k]-a)*share
            out[k] = round(value) if isinstance(self.target[k], int) else value
        # Wall time needs no report to advance while the run is alive.
        if out.get('elapsed_seconds') is not None and out.get('phase') not in ENDED and '_source_mtime' in out:
            out['elapsed_seconds'] += max(0., time.time()-out['_source_mtime'])
        return out


def watch(path, once=False, heartbeat=None):
    path=Path(path)
    tty=sys.stdout.isatty()
    # Windows terminals support VT even when their parent exported TERM=dumb.
    # Keep the override local to this renderer (including --attach).
    environ=dict(os.environ)
    if tty and os.name=='nt':
        environ['TERM']='xterm-256color'
    console=Console(legacy_windows=False,force_terminal=tty,force_interactive=tty,
                    color_system='truecolor' if tty else None,no_color=not tty,
                    _environ=environ)
    parsed={}
    def load(file):
        """(text, JSON, stat). stat() is the per-frame probe; the file is opened only after its
        mtime moved, or twice a second while it is fresh: two writes inside one timestamp tick
        (a run's last epoch and its final state) would otherwise hide the second one."""
        stat=file.stat()
        stamp,text,value,opened=parsed.get(file,(None,None,None,0.))
        fresh=time.time()-stat.st_mtime<2 and time.monotonic()-opened>=.5
        if stamp!=stat.st_mtime_ns or fresh:
            text=file.read_text(encoding='utf-8-sig')
            parsed[file]=(stat.st_mtime_ns,text,json.loads(text),time.monotonic())
        return parsed[file][1],parsed[file][2],stat
    def read():
        """(identity of what is on disk, state); (None, None) while a writer holds the file."""
        source=path
        try:
            text,value,stat=load(path)
            # A source manifest lets the launcher retarget the running window.
            if 'status_path' in value:
                source=Path(value['status_path'])
                text,value,stat=load(source)
            return (str(source),text),dict(value,_source_mtime=stat.st_mtime,_source_name=source.name)
        except FileNotFoundError:
            return (str(source),None),dict(_source_name=source.name)
        except (PermissionError,json.JSONDecodeError):
            return None,None
    # A pipe has no cursor or screen to replace. Never turn it into an endless
    # stream of full dashboards; callers can request another snapshot explicitly.
    if once or not tty:
        # Half-block card edges are painted with colour; without colour they would print as bars.
        console.print(render(read()[1] or {},width=console.width,edges=tty))
        return
    frame_seconds=MonitorConfig().frame_seconds
    # Full-screen replacement must use the alternate buffer. Cursor-up redraws
    # cannot erase rows that have already overflowed into terminal scrollback.
    # OSC 11 extends the panel colour into the window padding; OSC 111 gives it back.
    console.file.write(f'\x1b]11;{BASE}\x07')
    try:
        with Live(console=console,screen=True,auto_refresh=False,vertical_overflow='crop') as live:
            glide,identity,tick,beat=Glide(),None,0,0.
            while True:
                now=time.monotonic()
                found,state=read()
                if found is not None and found!=identity:
                    identity=found
                    glide.feed(state,now)
                live.update(render(glide.at(now),rows=console.height,width=console.width,tick=tick,
                                   interval=frame_seconds),refresh=True)
                if heartbeat and now-beat>=.5:
                    beat=now
                    heartbeat()
                tick+=1
                time.sleep(frame_seconds)
    finally:
        console.file.write('\x1b]111\x07')
        console.file.flush()

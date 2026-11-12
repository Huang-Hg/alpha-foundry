"""Where the evaluation service lives and what its files are (local runtime state under tmp/service/)."""
from __future__ import annotations

from foundry.config import ROOT

HOST = '127.0.0.1'
STATE = ROOT / 'tmp' / 'service'
PORTFILE = STATE / 'port.json'           # {port, pid, started, python} locally, or {host, port, ...} when attached remotely
LOCKFILE = STATE / 'start.lock'
LOGFILE = STATE / 'service.log'
STATUSFILE = STATE / 'status.json'      # dashboard feed

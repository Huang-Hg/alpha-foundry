"""Publishing a file that another process may be reading at the same moment (status files, the library)."""
from __future__ import annotations

import os
import time
from pathlib import Path


def publish(path: Path, content: bytes) -> None:
    """Replace `path` atomically: a reader sees the old file or the new one, never a partial write."""
    temp = path.with_suffix(f'{path.suffix}.{os.getpid()}.tmp')
    temp.write_bytes(content)
    while True:
        try:
            os.replace(temp, path)
            return
        except PermissionError:      # Windows: a reader has the file open right now
            time.sleep(0.02)

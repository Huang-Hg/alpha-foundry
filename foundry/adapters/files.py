"""Publishing a file that another process may be reading at the same moment (status files, the library)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish(path: Path, content: bytes) -> None:
    """Replace `path` atomically: a reader sees the old file or the new one, never a partial write."""
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

"""Library storage adapters with atomic compare-and-swap publication."""
from dataclasses import asdict
import json
import os
from pathlib import Path
from foundry.adapters.files import publish
from foundry.adapters.serialization import source_from_json
from foundry.portfolio.library import Library, Member, PROTOCOL
from foundry.research_types import ContextConflict


def member_from_json(record):
    return Member(source_from_json(record['source']), record['weight'])


class FileLibraryStore:
    def __init__(self, path: Path):
        self.path = path

    def read(self, study):
        if not self.path.exists():
            return Library((), study)
        record = json.loads(self.path.read_bytes())
        if record['protocol'] != PROTOCOL:
            raise ValueError('library protocol differs; rebuild the library')
        if record['study'] != asdict(study):
            raise ContextConflict('library belongs to a different study')
        return Library(tuple(member_from_json(m) for m in record['members']), study)

    def commit(self, expected, replacement):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix('.lock')
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        try:
            if self.read(expected.study).version != expected.version:
                raise ContextConflict('portfolio changed on disk')
            payload = dict(protocol=PROTOCOL, study=asdict(replacement.study), members=[asdict(m) for m in replacement.members])
            publish(self.path, json.dumps(payload, ensure_ascii=False, indent=1).encode())
        finally:
            lock.unlink()


class MemoryLibraryStore:
    def __init__(self, library):
        self.library = library

    def read(self, study):
        if study != self.library.study:
            raise ContextConflict('library belongs to a different study')
        return self.library

    def commit(self, expected, replacement):
        if self.library.version != expected.version:
            raise ContextConflict('portfolio changed')
        self.library = replacement

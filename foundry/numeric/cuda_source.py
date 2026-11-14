"""Shared CUDA arithmetic source, read only during explicit preparation."""
from pathlib import Path


def paired_source():
    return Path(__file__).with_name('float_pair.cuh').read_text(encoding='utf-8')

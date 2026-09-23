"""C ABI glue. No Python sampling, gradients, reward math or parameter loops."""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DOUBLE = ct.POINTER(ct.c_double)
_LIB = None


def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    source = Path(__file__).with_name('native.cpp')
    digest = hashlib.sha256(source.read_bytes() + sys.platform.encode()).hexdigest()[:16]
    directory = ROOT / 'cache' / 'q25_native_search' / digest
    binary = directory / ('search.dll' if os.name == 'nt' else 'search.so')
    if not binary.exists():
        compiler = os.environ.get('Q25_CXX') or shutil.which('g++')
        if not compiler and os.name == 'nt':
            compiler = 'C:/msys64/ucrt64/bin/g++.exe'
        if not compiler:
            raise RuntimeError('C++ compiler required; set Q25_CXX')
        directory.mkdir(parents=True, exist_ok=True)
        temp = binary.with_name(f'search-{os.getpid()}{binary.suffix}')
        args = [compiler, '-std=c++17', '-O3', '-shared', '-Wall', '-Wextra', str(source), '-o', str(temp)]
        args += ['-static', '-static-libgcc', '-static-libstdc++'] if os.name == 'nt' else ['-fPIC']
        built = subprocess.run(args, capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)
        os.replace(temp, binary)
    lib = ct.CDLL(str(binary))
    signatures = {
        'q25_error': ([], ct.c_char_p),
        'q25_create': ([ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_uint64], ct.c_void_p),
        'q25_destroy': ([ct.c_void_p], None),
        'q25_sample': ([ct.c_void_p, ct.c_int, ct.c_int, DOUBLE, ct.c_int], ct.c_char_p),
        'q25_train': ([ct.c_void_p, DOUBLE, ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double], ct.c_char_p),
        'q25_gradient_check': ([ct.c_void_p, DOUBLE, ct.c_int], ct.c_char_p),
        'q25_checkpoint': ([ct.c_void_p, ct.c_char_p, ct.c_int], ct.c_char_p),
        'q25_cem_sample': ([DOUBLE, DOUBLE, ct.c_int, ct.c_int, ct.c_uint64, DOUBLE], ct.c_int),
        'q25_cem_update': ([DOUBLE, DOUBLE, ct.c_int, ct.c_int, ct.c_int, DOUBLE, DOUBLE], ct.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    _LIB = lib
    return lib


def array(values):
    return (ct.c_double * len(values))(*values)


def checked(value):
    if not value:
        raise RuntimeError(library().q25_error().decode())
    return json.loads(value)


def grammar(fields, window_caps=(730, 365)):
    """Protocol: token, output type, AST cost, child count, child types."""
    rows = [f'{name} 0 1 0' for name in sorted(fields)]
    for name in ('sign', 'abs', 'neg', 'log', 'sqrt', 'tanh', 's_log_1p'):
        rows.append(f'{name} 0 2 1 0')
    for name in ('add', 'sub', 'mul', 'div', 'max', 'min', 'gt', 'lt'):
        rows.append(f'{name} 0 3 2 0 0')
    for name in ('gt', 'lt'):
        rows.append(f'{name} 0 3 2 0 1')
    for name in ('add_const', 'mul_const', 'pow_const'):
        rows.append(f'{name} 0 2 2 0 {1 if name == "add_const" else 4}')
    for name in ('ts_mean', 'ts_std', 'ts_min', 'ts_max', 'ts_delta', 'ts_ref',
                 'ts_slope', 'ts_zscore', 'ts_chanpos', 'ts_arg_min', 'ts_arg_max', 'ts_ddown'):
        rows.append(f'{name} 0 3 2 0 2')
    rows.append('ts_rank 0 3 2 0 3')
    for name in ('ts_corr', 'ts_cov'):
        rows.append(f'{name} 0 5 3 0 0 5')
    windows = (2, 3, 5, 7, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240, 365, 730)
    rows += [f'{w} 2 0 0' for w in windows if w <= window_caps[0]]
    rows += [f'{w} 5 0 0' for w in windows if w <= window_caps[1]]
    rows += [f'{k} 1 1 0' for k in ('-2', '-1', '-0.5', '-0.2', '-0.1', '-0.05', '0',
                                     '0.05', '0.1', '0.2', '0.5', '0.8', '1', '2')]
    return '\n'.join(rows)


class Policy:
    def __init__(self, fields, seed=1, max_tokens=13, max_cost=25, max_ts_depth=2, window_caps=(730, 365)):
        self.spec = grammar(fields, window_caps)
        self.lib = library()
        self.handle = self.lib.q25_create(self.spec.encode(), max_tokens, max_cost, max_ts_depth, seed)
        if not self.handle:
            raise RuntimeError(self.lib.q25_error().decode())

    def close(self):
        if self.handle:
            self.lib.q25_destroy(self.handle)
            self.handle = None

    def sample(self, count, probes, context, mode='gfn'):
        if len(context) != 16 or mode not in ('gfn', 'random'):
            raise ValueError('Expected 16 context values and gfn/random mode')
        return checked(self.lib.q25_sample(self.handle, count, probes, array(context), mode == 'random'))

    def train(self, rewards, epoch, steps=4, lr=.002, opd=.1):
        return checked(self.lib.q25_train(self.handle, array(rewards), len(rewards), epoch, steps, lr, opd))

    def gradient_check(self, rewards):
        return checked(self.lib.q25_gradient_check(self.handle, array(rewards), len(rewards)))

    def checkpoint(self, path, load=False):
        checked(self.lib.q25_checkpoint(self.handle, os.fsencode(path), load))


class CEM:
    def __init__(self, initial_logits):
        self.n = len(initial_logits)
        self.mean, self.sd = array(initial_logits), array([.5] * self.n)

    def sample(self, count, seed):
        output = (ct.c_double * (count * self.n))()
        if not library().q25_cem_sample(self.mean, self.sd, self.n, count, seed, output):
            raise RuntimeError(library().q25_error().decode())
        self.output, self.count = output, count
        return memoryview(output).cast('B')

    def update(self, scores, elite):
        if len(scores) != self.count:
            raise ValueError('CEM score count mismatch')
        if not library().q25_cem_update(self.output, array(scores), self.n, self.count, elite, self.mean, self.sd):
            raise RuntimeError(library().q25_error().decode())

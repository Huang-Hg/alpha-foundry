"""Executable ownership boundaries; CPU-only and no market data required."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'foundry'


def sources():
    return {'.'.join(p.relative_to(ROOT).with_suffix('').parts): p
            for p in PACKAGE.rglob('*.py')}


def imports(module, tree, names):
    """Resolve all imports, including function-local and relative forms."""
    edges = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = module.split('.')[:-node.level] if node.level else []
            base = '.'.join(prefix + ([node.module] if node.module else []))
            edges.add(base)
            edges.update(f'{base}.{a.name}' for a in node.names if f'{base}.{a.name}' in names)
        elif isinstance(node, ast.Call):
            dynamic = isinstance(node.func, ast.Name) and node.func.id == '__import__'
            dynamic |= isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'
            if dynamic:
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise AssertionError(f'{module}:{node.lineno}: dynamic imports must name a literal module')
                edges.add(node.args[0].value)
    return edges


def belongs(module, roots):
    return any(module == root or module.startswith(root + '.') for root in roots)


def import_time_nodes(node):
    """Function bodies are delayed; decorators, defaults and class bodies are not."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        for child in getattr(node, 'decorator_list', []):
            yield from import_time_nodes(child)
        for child in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            yield from import_time_nodes(child)
        return
    for child in ast.iter_child_nodes(node):
        yield from import_time_nodes(child)


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = sources()
        cls.trees = {name: ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
                     for name, path in cls.files.items()}
        cls.dependencies = {name: imports(name, tree, cls.files) for name, tree in cls.trees.items()}

    def test_source_dependencies_are_acyclic(self):
        visited, active = set(), []

        def visit(name):
            self.assertNotIn(name, active, ' -> '.join(active + [name]))
            if name in visited:
                return
            active.append(name)
            for target in sorted(self.dependencies[name] & self.files.keys()):
                visit(target)
            active.pop()
            visited.add(name)

        for name in sorted(self.files):
            visit(name)

    def test_domain_usecase_and_monitor_dependencies(self):
        domain = ('foundry.core', 'foundry.expr', 'foundry.market', 'foundry.portfolio',
                  'foundry.candidates', 'foundry.perpetual_types')
        concrete = ('foundry.adapters', 'foundry.runtime', 'foundry.engine', 'foundry.service',
                    'foundry.search', 'foundry.ops.cuda', 'foundry.backtest.cuda',
                    'foundry.backtest.perpetual_cpu', 'foundry.backtest.perpetual_cuda',
                    'foundry.numeric.cuda',
                    'cupy', 'pandas', 'pyarrow', 'socket', 'paramiko')
        for source, targets in self.dependencies.items():
            with self.subTest(module=source):
                if belongs(source, domain):
                    self.assertFalse([t for t in targets if belongs(t, concrete)])
                if source == 'foundry.research':
                    self.assertFalse([t for t in targets if belongs(t, concrete)])
                if belongs(source, ('foundry.monitor',)):
                    self.assertFalse([t for t in targets if belongs(t, domain +
                                     ('foundry.engine', 'foundry.research', 'foundry.search', 'foundry.ops', 'foundry.backtest', 'cupy'))])
                if source in ('foundry.contracts', 'foundry.candidates'):
                    self.assertFalse([t for t in targets if t.startswith('foundry.')])

    def test_import_does_not_initialize_devices_or_read_configuration(self):
        io = {'open', 'read', 'read_text', 'read_bytes', 'write_text', 'write_bytes', 'mkdir',
              'read_parquet', 'read_table', 'read_csv', 'rglob', 'glob', 'scandir', 'listdir',
              'socket', 'create_connection', 'urlopen'}
        gpu = {'RawModule', 'RawKernel', 'ElementwiseKernel', 'ReductionKernel', 'getDeviceProperties',
               'memGetInfo', 'get_default_memory_pool', 'get_default_pinned_memory_pool', 'Device', 'Stream'}
        for name, tree in self.trees.items():
            cupy_aliases = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                            for alias in node.names if alias.name == 'cupy'}
            for node in import_time_nodes(tree):
                if not isinstance(node, ast.Call):
                    continue
                with self.subTest(module=name, line=node.lineno):
                    if isinstance(node.func, ast.Attribute):
                        self.assertNotIn(node.func.attr, io | gpu)
                        self.assertFalse(isinstance(node.func.value, ast.Name) and node.func.value.id in cupy_aliases,
                                         'CuPy calls must be in an explicit resource lifecycle')
                    if isinstance(node.func, ast.Name):
                        self.assertNotIn(node.func.id, {'open', 'load_config', 'prepare_backend'})

    def test_lightweight_imports_need_neither_gpu_nor_dataset(self):
        modules = [name for name in sorted(self.files) if belongs(name,
                   ('foundry.contracts', 'foundry.candidates', 'foundry.perpetual_types', 'foundry.core', 'foundry.expr', 'foundry.market',
                    'foundry.research_types', 'foundry.portfolio.library', 'foundry.service.schema', 'foundry.ops.registry'))]
        script = r'''
import builtins, importlib, json, pathlib, sys
import numpy, jsonschema
from unittest.mock import patch
original = builtins.__import__
def controlled_import(name, *args, **kwargs):
    if name.split('.')[0] in ('cupy', 'paramiko'):
        raise AssertionError('lightweight import reached ' + name)
    return original(name, *args, **kwargs)
def no_io(*args, **kwargs):
    raise AssertionError('import attempted external file IO')
with patch('builtins.__import__', controlled_import), \
     patch('builtins.open', no_io), \
     patch.object(pathlib.Path, 'read_text', no_io), \
     patch.object(pathlib.Path, 'read_bytes', no_io), \
     patch.object(pathlib.Path, 'open', no_io), \
     patch.object(pathlib.Path, 'rglob', no_io):
    for name in json.loads(sys.argv[1]):
        importlib.import_module(name)
assert 'cupy' not in sys.modules
'''
        import json
        result = subprocess.run([sys.executable, '-c', script, json.dumps(modules)], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cuda_module_import_needs_no_device_or_kernel_files(self):
        script = r'''
import builtins, importlib, pathlib, sys, types
import numpy
from unittest.mock import patch

def prohibited(*args, **kwargs):
    raise AssertionError('CUDA import attempted resource access')

cupy = types.ModuleType('cupy')
cupy.float64 = numpy.float64
cupy.__getattr__ = prohibited
sys.modules['cupy'] = cupy
with patch('builtins.open', prohibited), \
     patch.object(pathlib.Path, 'read_text', prohibited), \
     patch.object(pathlib.Path, 'open', prohibited):
    importlib.import_module('foundry.ops.cuda')
    importlib.import_module('foundry.backtest.cuda')
    importlib.import_module('foundry.backtest.perpetual_cuda')
    importlib.import_module('foundry.numeric.cuda')
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

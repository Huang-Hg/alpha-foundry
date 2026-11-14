"""Failed startup releases every resource already acquired; no market data or GPU."""
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

from foundry import runtime
from foundry.backtest import cpu
from foundry.config import ExecutionConfig, load_config
from foundry.contracts import Study


class RuntimeLifecycleTests(unittest.TestCase):
    def test_missing_dataset_closes_backend_executor_and_resource_scope(self):
        settings = replace(load_config(), execution=ExecutionConfig('cpu'))
        study = Study('us_equity', settings.us_equity.start, settings.us_equity.end)
        resources = runtime.HostResources()
        backend = runtime.prepare_backend('cpu')
        close_backend = Mock(wraps=backend.close)
        backend = replace(backend, close=close_backend)
        executor = cpu.prepare()
        with patch.object(runtime, 'HostResources', return_value=resources), \
             patch.object(runtime, 'prepare_backend', return_value=backend), \
             patch.object(cpu, 'prepare', return_value=executor), \
             patch.object(executor, 'close', wraps=executor.close) as close_executor, \
             patch.object(resources, 'close', wraps=resources.close) as close_resources, \
             patch.object(runtime, 'load_data', side_effect=FileNotFoundError('missing daily artifact')):
            with self.assertRaisesRegex(FileNotFoundError, 'missing daily artifact'):
                runtime.open_runtime(settings, study)
            close_backend.assert_called_once_with()
            close_executor.assert_called_once_with()
            close_resources.assert_called_once_with()
        self.assertEqual(resources.streams, ())

    def test_backend_prepare_failure_closes_only_acquired_resources(self):
        settings = replace(load_config(), execution=ExecutionConfig('cpu'))
        study = Study('us_equity', settings.us_equity.start, settings.us_equity.end)
        resources = runtime.HostResources()
        with patch.object(runtime, 'HostResources', return_value=resources), \
             patch.object(runtime, 'prepare_backend', side_effect=RuntimeError('backend preparation failed')), \
             patch.object(cpu, 'prepare') as prepare_executor, \
             patch.object(resources, 'close', wraps=resources.close) as close_resources:
            with self.assertRaisesRegex(RuntimeError, 'backend preparation failed'):
                runtime.open_runtime(settings, study)
            prepare_executor.assert_not_called()
            close_resources.assert_called_once_with()
        self.assertEqual(resources.streams, ())


if __name__ == '__main__':
    unittest.main()

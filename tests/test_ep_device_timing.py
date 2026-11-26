"""Contract tests for example sampler wall-time accounting."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('ep_timing', Path(__file__).resolve().parents[1] / 'example/ep_mnist/device_timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def test_sampler_accounting_preserves_result_and_failure(monkeypatch):
    ticks = iter([1., 3., 4., 7.])
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: next(ticks))
    class Solver:
        def solve(self, matrix):
            if matrix is None:
                raise ValueError('failed')
            return matrix
    sampler = timing.SamplerTimer(Solver())
    expected = object()
    assert sampler.solve(expected) is expected
    with pytest.raises(ValueError):
        sampler.solve(None)
    assert sampler.calls == 2
    assert sampler.seconds == 5.


def test_cuda_request_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(timing.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='unavailable'):
        timing.select_device('cuda')
    assert str(timing.select_device('cpu')) == 'cpu'

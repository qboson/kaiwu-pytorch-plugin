"""Device selection and host wall-clock accounting for the MNIST example."""
import time
import torch


class SamplerTimer:
    """Measure synchronous CPU sampler calls without changing their results."""
    def __init__(self, sampler):
        self.sampler = sampler
        self.seconds = 0.0
        self.calls = 0

    def solve(self, matrix):
        started = time.perf_counter()
        try:
            return self.sampler.solve(matrix)
        finally:
            self.seconds += time.perf_counter() - started
            self.calls += 1


def select_device(value):
    """Reject unavailable devices instead of silently falling back."""
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu, cuda, or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError('CUDA device index is unavailable')
    return device


def synchronize(model):
    """Wait for model operations before reading a phase timer."""
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

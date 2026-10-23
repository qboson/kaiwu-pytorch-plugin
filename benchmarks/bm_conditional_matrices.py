"""CPU matrix-construction benchmark; does not invoke any SDK solver."""
import json
import pathlib
import statistics
import sys
import time

import kaiwu
import numpy as np
import torch

source = pathlib.Path(__file__).resolve().parents[1] / 'src/kaiwu'
kaiwu.__path__.insert(0, str(source))
from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine


def legacy_matrix(model, visible):
    with torch.no_grad():
        q = model.symmetrized_quadratic_coef()
        n = visible.numel()
        hh = q[n:, n:]
        bias = q[n:, :n] @ visible + model.linear_bias[n:]
        matrix = torch.zeros((len(bias) + 1, len(bias) + 1), dtype=bias.dtype)
        matrix[:-1, :-1] = hh / 8
        field = bias / 4 + hh.sum(0) / 8
        matrix[:-1, -1] = field
        matrix[-1, :-1] = field
        return matrix.numpy()


def time_call(call):
    call()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    torch.set_num_threads(1)
    rows = []
    for nodes, batch, hidden in [(128, 32, 16), (512, 64, 16), (512, 128, 16)]:
        timings = []
        for seed in [7, 17, 27]:
            torch.manual_seed(seed)
            model = BoltzmannMachine(nodes, device='cpu')
            visible = torch.rand(batch, nodes - hidden)
            def legacy():
                return [legacy_matrix(model, row) for row in visible]
            def cached():
                terms = model._conditional_ising_terms(visible.shape[1])
                return [model._hidden_to_ising_matrix(row, conditional_terms=terms) for row in visible]
            assert all(np.array_equal(a, b) for a, b in zip(legacy(), cached()))
            timings.append({'seed': seed, 'legacy_ms': time_call(legacy), 'cached_ms': time_call(cached)})
        rows.append({'nodes': nodes, 'batch': batch, 'hidden': hidden,
                     'legacy_ms': statistics.median(t['legacy_ms'] for t in timings),
                     'cached_ms': statistics.median(t['cached_ms'] for t in timings),
                     'seed_results': timings})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()

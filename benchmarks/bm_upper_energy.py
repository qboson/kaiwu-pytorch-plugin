"""Full BM energy CPU timing and operator allocations, without a sampler."""
import json
from pathlib import Path
import statistics
import time

import kaiwu
import torch

kaiwu.__path__.insert(0, str(Path(__file__).resolve().parents[1] / "src/kaiwu"))
from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine  # pylint: disable=wrong-import-position


def legacy(model, states):
    """Original symmetric energy evaluation."""
    return -states @ model.linear_bias - 0.5 * torch.sum(
        states.matmul(model.symmetrized_quadratic_coef()) * states, dim=-1,
    )


def measure(model, states, operation, backward):
    """Warm up, then measure five complete forward or forward/backward calls."""
    timings = []
    for repeat in range(6):
        model.zero_grad(set_to_none=True)
        states.grad = None
        start = time.perf_counter()
        with torch.set_grad_enabled(backward):
            output = operation(model, states)
            if backward:
                output.sum().backward()
        elapsed = (time.perf_counter() - start) * 1000
        if repeat:
            timings.append(elapsed)
    return statistics.median(timings)


torch.set_num_threads(1)
results = []
for nodes, batch in [(256, 1), (1024, 16), (2048, 16)]:
    row = {"nodes": nodes, "batch": batch}
    for backward in (False, True):
        old_times, changed_times = [], []
        for seed in (7, 17, 27):
            torch.manual_seed(seed)
            model = BoltzmannMachine(nodes, device="cpu")
            states = torch.rand(batch, nodes, requires_grad=True)
            torch.testing.assert_close(model(states), legacy(model, states), rtol=1e-4, atol=1e-4)
            old_times.append(measure(model, states, legacy, backward))
            changed_times.append(measure(model, states, lambda m, x: m(x), backward))
        tag = "forward_backward" if backward else "no_grad_forward"
        row[tag] = {"legacy_ms": statistics.median(old_times),
                    "optimized_ms": statistics.median(changed_times)}
    results.append(row)
allocations = {}
for name, operation in [("legacy", legacy), ("optimized", lambda m, x: m(x))]:
    with torch.no_grad(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
    ) as profile:
        operation(model, states)
    allocations[name] = {
        event.key: {"calls": event.count, "self_cpu_memory_bytes": event.self_cpu_memory_usage}
        for event in profile.key_averages() if event.key in ("aten::triu", "aten::add")
    }
print(json.dumps({"timings": results, "last_no_grad_operator_allocations": allocations}, indent=2))

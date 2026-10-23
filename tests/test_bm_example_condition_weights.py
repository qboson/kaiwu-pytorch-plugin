"""BM examples must weight observations independently of solver row counts."""

import runpy
import sys
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

# The examples are repository files, outside the installed src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example.bm_generation import trainer
from kaiwu.torch_plugin import BoltzmannMachine


def binary_states(width):
    return torch.tensor(list(product((0.0, 1.0), repeat=width)))


class OfflineSampler:
    """Return prescribed legal Ising rows through the real BM sampling API."""

    def __init__(self, groups):
        self.groups = groups
        self.calls = {width: 0 for width in self.groups}
        self.order = []

    def solve(self, matrix):
        width = len(matrix) - 1
        states = self.groups[width][self.calls[width]]
        self.calls[width] += 1
        self.order.append(width)
        spins = 2 * states.numpy() - 1
        return np.column_stack((spins, np.ones(len(spins), dtype=np.float32)))


class InlinePool:
    """Replace only the process boundary; execute the actual example worker."""

    def __init__(self, processes):
        self.processes = processes

    def map(self, worker, parameters):
        return [worker(parameter) for parameter in parameters]

    def close(self):
        pass

    def join(self):
        pass


class RecordingSaver:
    def __init__(self):
        self.losses = []

    def save_info(self, *args):
        pass

    def output_loss(self, step, kl_divergence, ncl, objective):
        self.losses.append((kl_divergence, ncl, objective))


class TestBMExampleConditionWeights(unittest.TestCase):
    def assert_expected_training(self, result, observed, full_groups, input_groups,
                                 negative, alpha):
        """Use sufficient statistics, independently of the example's loss code."""
        full = [torch.cat((row.expand(len(group), -1), group), dim=1)
                for row, group in zip(observed, full_groups)]
        inputs = [torch.cat((row[:-1].expand(len(group), -1), group), dim=1)
                  for row, group in zip(observed, input_groups)]

        def moments(groups):
            first = sum(group.mean(0) for group in groups) / len(groups)
            second = sum((group[:, :, None] * group[:, None, :]).mean(0)
                         for group in groups) / len(groups)
            return first, second

        def gradient(positive, negative_groups):
            pos_first, pos_second = moments(positive)
            neg_first, neg_second = moments(negative_groups)
            return neg_first - pos_first, (neg_second - pos_second).triu(1)

        kl_bias, kl_quad = gradient(full, [negative])
        # Preserve the example's existing last-column clearing in this fix.
        full_ncl = [group.clone() for group in full]
        input_ncl = [group.clone() for group in inputs]
        for group in full_ncl + input_ncl:
            group[:, -1] = 0
        ncl_bias, ncl_quad = gradient(full_ncl, input_ncl)
        bias_grad = alpha * kl_bias + (1 - alpha) * ncl_bias
        quad_grad = alpha * kl_quad + (1 - alpha) * ncl_quad
        initial_bias = torch.tensor([0.25, -0.5, 0.75, -0.25])
        initial_quad = torch.arange(16).reshape(4, 4) / 16
        kl = (kl_bias * initial_bias).sum() + (kl_quad * initial_quad).sum()
        ncl = (ncl_bias * initial_bias).sum() + (ncl_quad * initial_quad).sum()
        model, losses = result
        torch.testing.assert_close(torch.tensor(losses),
                                   torch.stack((kl, ncl, alpha * kl + (1 - alpha) * ncl)))
        for parameter, initial, expected_grad in (
                (model.linear_bias, initial_bias, bias_grad),
                (model.quadratic_coef, initial_quad, quad_grad)):
            torch.testing.assert_close(parameter.grad, expected_grad)
            # First Adam step, with its default eps and zero weight decay.
            expected_parameter = initial - 0.1 * expected_grad / (expected_grad.abs() + 1e-8)
            torch.testing.assert_close(parameter, expected_parameter)

    def run_training(self, observed, full_groups=None, input_groups=None,
                     num_processes=1, alpha=0.25, negative=None):
        full_groups = full_groups or [binary_states(2) for _ in observed]
        input_groups = input_groups or [binary_states(3) for _ in observed]
        negative = binary_states(4) if negative is None else negative
        sampler = OfflineSampler({2: full_groups, 3: input_groups, 4: [negative]})
        saver = RecordingSaver()
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(trainer.mp, "Pool", InlinePool):
            case = trainer.Trainer([observed.clone()], saver, sampler,
                                   num_visible=2, num_hidden=2, num_output=1)
            with torch.no_grad():
                case.bm_net.linear_bias.copy_(
                    torch.tensor([0.25, -0.5, 0.75, -0.25]))
                case.bm_net.quadratic_coef.copy_(
                    torch.arange(16).reshape(4, 4) / 16)
            case.set_cost_parameter(alpha=alpha, beta=0.0)
            case.set_learning_parameters(0.1, 0.0, 0.0)
            case.train(max_steps=1, save_path="unused", num_processes=num_processes)
        self.assertEqual(sampler.calls, {2: len(observed), 3: len(observed), 4: 1})
        result = case.bm_net, saver.losses[0]
        self.assert_expected_training(result, observed, full_groups, input_groups,
                                      negative, alpha)
        return result

    def assert_same_training(self, first, second):
        first_model, first_losses = first
        second_model, second_losses = second
        torch.testing.assert_close(torch.tensor(first_losses), torch.tensor(second_losses))
        for first_param, second_param in zip(first_model.parameters(), second_model.parameters()):
            torch.testing.assert_close(first_param.grad, second_param.grad)
            torch.testing.assert_close(first_param, second_param)

    def test_repeating_full_condition_samples_preserves_training(self):
        observed = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ordinary = self.run_training(observed)
        repeated = self.run_training(
            observed, full_groups=[binary_states(2), binary_states(2).repeat(4, 1)])
        self.assert_same_training(ordinary, repeated)

    def test_repeating_input_condition_samples_preserves_training(self):
        observed = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ordinary = self.run_training(observed)
        repeated = self.run_training(
            observed, input_groups=[binary_states(3).repeat(4, 1), binary_states(3)])
        self.assert_same_training(ordinary, repeated)

    def test_phases_keep_independent_counts(self):
        observed = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        ordinary = self.run_training(observed)
        repeated = self.run_training(
            observed,
            full_groups=[binary_states(2), binary_states(2).repeat(4, 1)],
            input_groups=[binary_states(3).repeat(2, 1), binary_states(3)])
        self.assert_same_training(ordinary, repeated)

    def test_uneven_worker_chunks_keep_each_observation_weight(self):
        observed = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0],
                                 [1.0, 1.0], [1.0, 1.0]])
        full = [binary_states(2).repeat(count, 1) for count in (1, 4, 2, 4, 1)]
        inputs = [binary_states(3).repeat(count, 1) for count in (2, 1, 4, 1, 2)]
        # Nonzero gradients avoid amplifying float32 cancellation through Adam eps.
        negative = torch.zeros(1, 4)
        one_chunk = self.run_training(observed, full, inputs, negative=negative)
        uneven_chunks = self.run_training(observed, full, inputs, num_processes=2,
                                          negative=negative)
        self.assert_same_training(one_chunk, uneven_chunks)

    def test_equal_counts_preserve_flat_phase_objective(self):
        observed = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        self.run_training(observed)

    def test_repeated_observations_keep_their_empirical_frequency(self):
        observed = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        model, _ = self.run_training(observed, alpha=1.0)
        self.assertAlmostEqual(model.linear_bias.grad[0].item(), -1 / 6, places=6)

    def test_hidden_correlations_are_averaged_after_energy(self):
        observed = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        model, _ = self.run_training(
            observed,
            full_groups=[torch.tensor([[0.0, 0.0], [1.0, 1.0]]), binary_states(2)],
            alpha=1.0)
        self.assertAlmostEqual(model.quadratic_coef.grad[2, 3].item(), -0.125)

    def test_unequal_unique_solutions_and_negative_sample_counts(self):
        observed = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        full = [torch.tensor([[0.0, 0.0]]),
                torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])]
        inputs = [torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                  torch.tensor([[1.0, 0.0, 1.0]])]
        for group in full + inputs:
            self.assertEqual(len(group.unique(dim=0)), len(group))
        negative = torch.tensor([[0.0] * 4, [1.0] * 4, [1.0] * 4])
        self.run_training(observed, full, inputs, negative=negative)

    def test_worker_returns_counts_in_observation_order(self):
        observed = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        full = [torch.tensor([[0.0, 0.0]]),
                torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])]
        inputs = [torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                  torch.tensor([[1.0, 0.0, 1.0]])]
        sampler = OfflineSampler({2: full, 3: inputs})
        model = BoltzmannMachine(4, device="cpu")
        result = trainer.process_solve_graph_worker((model, sampler, observed, 1))
        self.assertEqual(tuple(result[2]), (1, 3))
        self.assertEqual(tuple(result[3]), (2, 1))
        for sampled, conditions, groups in ((result[0], observed, full),
                                            (result[1], observed[:, :-1], inputs)):
            expected = torch.cat([torch.cat((row.expand(len(group), -1), group), 1)
                                  for row, group in zip(conditions, groups)])
            torch.testing.assert_close(sampled, expected)
            self.assertFalse(sampled.requires_grad)
        self.assertEqual(sampler.order, [2, 2, 3, 3])

    def test_small_example_preserves_weights_when_a_group_is_repeated(self):
        observed = torch.tensor([[0, 0], [0, 1], [0, 0], [1, 0], [1, 1]])
        results = []
        for repeat in (1, 4):
            full = [binary_states(3) for _ in observed]
            full[-1] = full[-1].repeat(repeat, 1)
            sampler = OfflineSampler({3: full, 5: [binary_states(5)]})
            model = BoltzmannMachine(5, device="cpu")
            with torch.no_grad():
                model.linear_bias.copy_(torch.arange(5) / 4)
                model.quadratic_coef.copy_(torch.arange(25).reshape(5, 5) / 16)
            with patch("kaiwu.classical.SimulatedAnnealingOptimizer", return_value=sampler), \
                    patch("kaiwu.torch_plugin.BoltzmannMachine", return_value=model), \
                    patch.object(torch, "randint", return_value=observed), \
                    patch("builtins.print"):
                namespace = runpy.run_path(
                    str(Path(trainer.__file__).resolve().parents[1] / "run_bm.py"),
                    run_name="__main__")
            self.assertEqual(sampler.calls, {3: 5, 5: 1})
            results.append((model, (namespace["objective"].item(),)))
        self.assert_same_training(results[0], results[1])


if __name__ == "__main__":
    unittest.main()

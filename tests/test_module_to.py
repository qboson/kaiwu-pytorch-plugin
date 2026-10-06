import unittest

import torch

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from kaiwu.torch_plugin import BoltzmannMachine as BM
from kaiwu.torch_plugin import RestrictedBoltzmannMachine as RBM
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine as GBRBM


def make_models():
    """One instance of every AbstractBoltzmannMachine subclass."""
    return [
        ("BoltzmannMachine", BM(4)),
        ("RestrictedBoltzmannMachine", RBM(4, 3)),
        ("GaussianBernoulliRestrictedBoltzmannMachine", GBRBM(4, 3)),
    ]


class TestBoltzmannMachineTo(unittest.TestCase):
    """`to()` documents `dtype` and `non_blocking`, but dropped both.

    The override used `device=...` sentinels and forwarded only `device` to
    `nn.Module.to()`, so a dtype-only call passed the Ellipsis sentinel straight
    into torch:

        model.to(dtype=torch.float64)
        TypeError: to() received an invalid combination of arguments - got (ellipsis)

    A positional dtype was worse: `to(torch.float64)` bound the dtype to the
    `device` parameter, so it could never work at all.
    """

    def test_to_dtype_keyword_is_applied(self):
        """to(dtype=...) must convert the parameters, not raise."""
        with self.subTest("keyword dtype"):
            for name, model in make_models():
                with self.subTest(model=name):
                    model.to(dtype=torch.float64)
                    for pname, param in model.named_parameters():
                        self.assertEqual(
                            param.dtype,
                            torch.float64,
                            f"{name}.{pname} did not change dtype",
                        )

    def test_to_dtype_positional_is_applied(self):
        """to(torch.float64) must work; the dtype used to land on `device`."""
        with self.subTest("positional dtype"):
            for name, model in make_models():
                with self.subTest(model=name):
                    model.to(torch.float64)
                    for pname, param in model.named_parameters():
                        self.assertEqual(
                            param.dtype,
                            torch.float64,
                            f"{name}.{pname} did not change dtype",
                        )

    def test_device_and_dtype_together(self):
        """Both keyword arguments must be honoured at once."""
        with self.subTest("device + dtype"):
            for name, model in make_models():
                with self.subTest(model=name):
                    model.to(device=torch.device("cpu"), dtype=torch.float64)
                    self.assertEqual(model.linear_bias.dtype, torch.float64)
                    self.assertEqual(model.device, torch.device("cpu"))

    def test_device_only_still_works(self):
        """The already-working device path must not regress."""
        with self.subTest("device only"):
            for name, model in make_models():
                with self.subTest(model=name):
                    model.to(torch.device("cpu"))
                    self.assertEqual(model.device, torch.device("cpu"))
                    model.to("cpu")
                    self.assertEqual(model.linear_bias.device, torch.device("cpu"))

    def test_self_device_stays_a_torch_device(self):
        """self.device is documented as a torch.device and is fed to torch.zeros.

        A dtype-only call used to leave it as the Ellipsis sentinel, and a
        string device left it a bare str.
        """
        with self.subTest("self.device type"):
            for name, model in make_models():
                with self.subTest(model=name):
                    model.to(dtype=torch.float32)
                    self.assertIsInstance(
                        model.device,
                        torch.device,
                        f"{name}: self.device became {type(model.device).__name__}",
                    )
                    model.to("cpu")
                    self.assertIsInstance(model.device, torch.device)

    def test_to_with_no_arguments_is_a_noop(self):
        """to() with nothing must not clobber self.device."""
        with self.subTest("no arguments"):
            for name, model in make_models():
                with self.subTest(model=name):
                    before = model.device
                    model.to()
                    self.assertEqual(model.device, before)
                    self.assertIsInstance(model.device, torch.device)

    def test_non_blocking_is_accepted(self):
        """non_blocking is in the signature and must not raise."""
        with self.subTest("non_blocking"):
            for name, model in make_models():
                with self.subTest(model=name):
                    model.to("cpu", non_blocking=True)

    def test_model_still_usable_after_dtype_conversion(self):
        """A converted model must still compute, in the converted dtype."""
        with self.subTest("usable after to(dtype=...)"):
            rbm = RBM(4, 3)
            rbm.to(dtype=torch.float64)
            s_all = torch.zeros(2, rbm.num_visible + rbm.num_hidden, dtype=torch.float64)
            energy = rbm(s_all)
            self.assertEqual(tuple(energy.shape), (2,))
            self.assertEqual(energy.dtype, torch.float64)

    def test_gbrbm_dtype_attribute_tracks_parameters(self):
        """GBRBM allocates with `dtype=self.dtype` in 15 places.

        Once to(dtype=...) works, a stale self.dtype would make
        _to_ising_matrix build a float32 matrix for float64 parameters.
        """
        with self.subTest("self.dtype stays in sync"):
            model = GBRBM(4, 3)
            model.to(dtype=torch.float64)
            self.assertEqual(model.dtype, torch.float64)
            # get_ising_matrix() returns a numpy array, so compare dtype names.
            self.assertEqual(model.get_ising_matrix().dtype.name, "float64")

            model.to(dtype=torch.float32)
            self.assertEqual(model.dtype, torch.float32)
            self.assertEqual(model.get_ising_matrix().dtype.name, "float32")


if __name__ == "__main__":
    unittest.main()
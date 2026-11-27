"""Real checkpoint I/O tests for the simple QDEQ examples."""

from pathlib import Path
import pickle
import pickletools
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile

import torch

from example.qdeq.simple.checkpoint_io import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class CheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_path_and_string_preserve_tensors_and_metadata(self):
        path = self.directory / "checkpoint.pt"
        payload = {
            "state_dict": {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
            "epoch": 4,
            "metrics": {"loss": 0.125},
        }
        torch.save(payload, path)
        for argument in (path, str(path)):
            with self.subTest(path_type=type(argument).__name__):
                loaded = load_checkpoint(argument)
                self.assertEqual(set(loaded), set(payload))
                self.assertEqual(loaded["epoch"], 4)
                self.assertEqual(loaded["metrics"], payload["metrics"])
                weight = loaded["state_dict"]["weight"]
                torch.testing.assert_close(weight, payload["state_dict"]["weight"])
                self.assertEqual(weight.device.type, "cpu")

    def test_missing_path_names_training_script_and_checkpoint(self):
        path = self.directory / "not-trained.pt"
        for argument in (path, str(path)):
            with self.subTest(path_type=type(argument).__name__):
                with self.assertRaises(FileNotFoundError) as raised:
                    load_checkpoint(argument)
                self.assertIn(str(path), str(raised.exception))
                self.assertIn("simple_train_example.py", str(raised.exception))

    def test_cuda_tagged_checkpoint_loads_on_cpu(self):
        path = self.directory / "cuda-tagged.pt"
        # The child registers a supported serialization tagger for CPU storage.
        # It saves real tensor bytes tagged cuda:0 without allocating a CUDA
        # tensor; its registry changes cannot affect this process or other tests.
        fixture_code = textwrap.dedent('''
            import sys
            import torch

            torch.serialization.register_package(
                0,
                lambda storage: "cuda:0" if storage.device.type == "cpu" else None,
                lambda storage, location: None,
            )
            torch.save({"state_dict": {"weight": torch.arange(4)}, "epoch": 9}, sys.argv[1])
        ''')
        subprocess.run([sys.executable, "-c", fixture_code, str(path)], check=True)
        with zipfile.ZipFile(path) as archive:
            pickle_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
            strings = [arg for _, arg, _ in pickletools.genops(archive.read(pickle_name))]
        self.assertIn("cuda:0", strings)
        if torch.version.cuda is None:
            # The original direct load fails for this genuine device-tagged file.
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                torch.load(path)
        loaded = load_checkpoint(path)
        self.assertEqual(loaded["epoch"], 9)
        weight = loaded["state_dict"]["weight"]
        self.assertEqual(weight.device.type, "cpu")
        torch.testing.assert_close(weight, torch.arange(4))

    def test_corrupt_checkpoint_error_is_not_replaced_with_missing_guidance(self):
        path = self.directory / "corrupt.pt"
        path.write_bytes(b"not a checkpoint")
        with self.assertRaises(pickle.UnpicklingError) as direct:
            torch.load(path, map_location="cpu")
        with self.assertRaises(type(direct.exception)) as helper:
            load_checkpoint(path)
        self.assertEqual(str(helper.exception), str(direct.exception))

    def test_directory_error_is_preserved(self):
        with self.assertRaises(IsADirectoryError):
            load_checkpoint(self.directory)


if __name__ == "__main__":
    unittest.main()

"""Resolution parsing rejects empty lists without changing float parsing."""
import importlib
import importlib.util
import math
import unittest


class TestResolutionParsing(unittest.TestCase):
    observations = []

    @classmethod
    def setUpClass(cls):
        try:
            cls.benchmark = importlib.import_module(
                "example.qvae_cell.evaluate_benchmark"
            )
        except ModuleNotFoundError as error:
            if error.name in {"anndata", "scanpy"} and importlib.util.find_spec(
                error.name
            ) is None:
                raise unittest.SkipTest(
                    f"Example benchmark dependency {error.name!r} is not installed"
                ) from error
            raise

    def test_empty_resolution_lists_raise_parameter_error(self):
        for value in ("", " \t\n", ",", ",,,", " , \t, "):
            with self.subTest(value=value):
                error = None
                returned = None
                try:
                    returned = self.benchmark.parse_resolutions(value)
                except Exception as observed:
                    error = observed
                self.observations.append({
                    "input": value,
                    "outcome": "returned" if error is None else type(error).__name__,
                    "returned": returned,
                    "message": None if error is None else str(error),
                })
                self.assertIsInstance(error, ValueError)
                self.assertIn("resolutions", str(error).lower())

    def test_nonempty_lists_preserve_float_values_and_token_handling(self):
        cases = (
            ("0.2,0.4,0.8", [0.2, 0.4, 0.8]),
            (" 0.2 ,\t0.4\n", [0.2, 0.4]),
            (",0.2,,0.4, ", [0.2, 0.4]),
            ("-1,0,2e-1", [-1.0, 0.0, 0.2]),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                actual = self.benchmark.parse_resolutions(value)
                self.assertEqual(actual, expected)
                self.assertTrue(all(type(item) is float for item in actual))

    def test_nonfinite_tokens_retain_float_semantics(self):
        actual = self.benchmark.parse_resolutions("nan,inf,-inf")
        self.assertEqual(len(actual), 3)
        self.assertTrue(math.isnan(actual[0]))
        self.assertEqual(actual[1:], [math.inf, -math.inf])

    def test_malformed_number_retains_value_error(self):
        with self.assertRaises(ValueError):
            self.benchmark.parse_resolutions("0.2,invalid")


if __name__ == "__main__":
    unittest.main()

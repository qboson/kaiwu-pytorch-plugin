"""Regression tests for the DPLM count-distribution convention."""

import importlib.util
import math
import unittest


class CountDistributionTests(unittest.TestCase):
    """Exercise count metrics through their normal example-package imports."""

    @classmethod
    def setUpClass(cls):
        try:
            from example.qdiffusion.dplm.utils import metrics
        except ModuleNotFoundError as exc:
            missing = exc
            # The builder retries a failed relative model import as "models".
            if exc.name == "models" and isinstance(
                exc.__context__, ModuleNotFoundError
            ):
                missing = exc.__context__
            if (
                missing.name in {"omegaconf", "transformers"}
                and importlib.util.find_spec(missing.name) is None
            ):
                raise unittest.SkipTest(
                    f"Optional DPLM dependency {missing.name!r} is not installed"
                ) from exc
            raise
        cls.metrics = metrics

    def test_one_sided_zero_totals(self):
        pairs = [
            ({}, {"A": 1}),
            ({"A": 0}, {"A": 3}),
            ({"A": 0, "B": 0}, {"C": 2}),
        ]
        for reference, candidate in pairs:
            for left, right in ((reference, candidate), (candidate, reference)):
                with self.subTest(left=left, right=right):
                    before = (left.copy(), right.copy())
                    self.assertAlmostEqual(
                        self.metrics.jsd_from_counts(left, right), math.log(2)
                    )
                    self.assertEqual((left, right), before)

    def test_both_zero_totals(self):
        for left, right in (
            ({}, {}),
            ({}, {"A": 0}),
            ({"A": 0}, {}),
            ({"A": 0}, {"B": 0}),
        ):
            with self.subTest(left=left, right=right):
                self.assertEqual(self.metrics.jsd_from_counts(left, right), 0.0)

    def test_real_counting_callers(self):
        self.assertEqual(
            self.metrics.token_distribution(iter(["ABA", "B"])), {"A": 2, "B": 2}
        )
        self.assertEqual(
            self.metrics.kmer_distribution(iter(["AB", "ABA"]), 2), {"AB": 2, "BA": 1}
        )
        pairs = [
            (
                self.metrics.kmer_distribution(iter(["AB", "A"]), 3),
                self.metrics.kmer_distribution(iter(["ABC"]), 3),
            ),
            (
                self.metrics.token_distribution(iter(["", ""])),
                self.metrics.token_distribution(iter(["A"])),
            ),
        ]
        for empty_counts, populated_counts in pairs:
            with self.subTest(populated_counts=populated_counts):
                self.assertEqual(empty_counts, {})
                self.assertAlmostEqual(
                    self.metrics.jsd_from_counts(empty_counts, populated_counts), math.log(2)
                )

    def test_non_degenerate_controls(self):
        cases = [
            ({"A": 2, "B": 1}, {"A": 2, "B": 1}, 0.0),
            ({"A": 2, "B": 1}, {"A": 6, "B": 3}, 0.0),
            ({"A": 1}, {"B": 1}, math.log(2)),
            ({"A": 1}, {"A": 1, "B": 1}, 0.75 * math.log(4 / 3)),
        ]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                before = (left.copy(), right.copy())
                self.assertAlmostEqual(self.metrics.jsd_from_counts(left, right), expected)
                self.assertAlmostEqual(self.metrics.jsd_from_counts(right, left), expected)
                scaled_left = {key: 7 * value for key, value in left.items()}
                scaled_right = {key: 3 * value for key, value in right.items()}
                self.assertAlmostEqual(
                    self.metrics.jsd_from_counts(scaled_left, scaled_right), expected
                )
                self.assertEqual((left, right), before)


if __name__ == "__main__":
    unittest.main()

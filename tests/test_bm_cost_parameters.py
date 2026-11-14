"""Behavioral tests for the real stdlib cost-parameter validation component."""
from decimal import Decimal
from fractions import Fraction
import sys
import unittest

from example.bm_generation.cost_parameters import validate_cost_parameters


class TestCostParameters(unittest.TestCase):
    def assert_costs(self, alpha, beta, expected):
        result = validate_cost_parameters(alpha, beta)
        self.assertEqual(result, expected)
        self.assertIs(type(result["alpha"]), float)
        self.assertIs(type(result["beta"]), float)

    def test_documented_cost_pair(self):
        self.assert_costs(0.5, 10, {"alpha": 0.5, "beta": 10.0})

    def test_normalizes_valid_inputs_without_an_upper_weight_limit(self):
        cases = [
            (0, 0, {"alpha": 0.0, "beta": 0.0}),
            (-0.0, 0.0, {"alpha": 0.0, "beta": 0.0}),
            (2, 3, {"alpha": 2.0, "beta": 3.0}),
            (Decimal("0.25"), Fraction(3, 2), {"alpha": 0.25, "beta": 1.5}),
            ("0.5", "10", {"alpha": 0.5, "beta": 10.0}),
            (True, False, {"alpha": 1.0, "beta": 0.0}),
            (sys.float_info.max, sys.float_info.min,
             {"alpha": sys.float_info.max, "beta": sys.float_info.min}),
        ]
        for alpha, beta, expected in cases:
            with self.subTest(alpha=alpha, beta=beta):
                self.assert_costs(alpha, beta, expected)

    def test_rejects_negative_and_nonfinite_values_in_either_field(self):
        invalid = [-1.0, -0.25, float("nan"), float("inf"), -float("inf")]
        for name in ("alpha", "beta"):
            for value in invalid:
                with self.subTest(parameter=name, value=value):
                    parameters = {"alpha": 0.5, "beta": 10}
                    parameters[name] = value
                    with self.assertRaisesRegex(ValueError, name + ".*finite non-negative"):
                        validate_cost_parameters(**parameters)

    def test_reports_conversion_failures_in_either_field(self):
        invalid = [None, "not a number", [], object(), complex(1, 2), 10**1000]
        for name in ("alpha", "beta"):
            for value in invalid:
                with self.subTest(parameter=name, value_type=type(value).__name__):
                    parameters = {"alpha": 0.5, "beta": 10}
                    parameters[name] = value
                    with self.assertRaisesRegex(ValueError, name + ".*finite non-negative"):
                        validate_cost_parameters(**parameters)

    def test_checks_values_after_float_conversion(self):
        for name in ("alpha", "beta"):
            for value in ("NaN", "+Inf", "-Inf"):
                with self.subTest(parameter=name, value=value):
                    parameters = {"alpha": 0.5, "beta": 10}
                    parameters[name] = value
                    with self.assertRaisesRegex(ValueError, name + ".*finite non-negative"):
                        validate_cost_parameters(**parameters)


if __name__ == "__main__":
    unittest.main()

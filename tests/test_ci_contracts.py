"""Regression checks for license prerequisites and CI acceptance evidence."""

from pathlib import Path
import runpy
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from conftest import initialize_kaiwu_license, reject_license_input


CHECKER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / ".github/scripts/check_ci_results.py")
)


@pytest.mark.parametrize("credentials", [{}, {"KAIWU_USER_ID": "user"},
                                        {"KAIWU_SDK_CODE": "code"},
                                        {"KAIWU_USER_ID": " ", "KAIWU_SDK_CODE": "code"}])
def test_required_license_rejects_missing_credentials(credentials):
    initializer = Mock()
    with pytest.raises(RuntimeError, match="require both"):
        initialize_kaiwu_license({"KAIWU_REQUIRE_LICENSE": "1", **credentials}, initializer)
    initializer.assert_not_called()


def test_local_cached_license_does_not_reactivate():
    initializer = Mock()
    initialize_kaiwu_license({}, initializer)
    initializer.assert_not_called()


def test_license_initialization_uses_explicit_credentials():
    initializer = Mock()
    initialize_kaiwu_license({"KAIWU_REQUIRE_LICENSE": "1", "KAIWU_USER_ID": "user",
                             "KAIWU_SDK_CODE": "code"}, initializer)
    initializer.assert_called_once_with("user", "code")


def test_license_failure_does_not_expose_sdk_output(capsys):
    def fail(*_args):
        print("private-example-code")
        raise ValueError("private-example-code")

    with pytest.raises(RuntimeError, match="initialization failed") as caught:
        initialize_kaiwu_license({"KAIWU_USER_ID": "user", "KAIWU_SDK_CODE": "code"}, fail)
    assert "private-example-code" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert capsys.readouterr().out == ""


def test_interactive_license_input_fails_immediately():
    with pytest.raises(RuntimeError, match="Interactive Kaiwu license input is disabled"):
        reject_license_input("Enter user ID")


def write_report(tmp_path, cases):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for classname, name, outcome in cases:
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        if outcome:
            ET.SubElement(case, outcome)
    report = tmp_path / "tests.xml"
    ET.ElementTree(root).write(report, encoding="utf-8")
    return report


def acceptance_cases():
    return [(classname, name, None)
            for classname, name in sorted(CHECKER["LICENSED_CASES"] | CHECKER["MNIST_CASES"])]


def test_ci_accepts_actual_solver_and_mnist_passes(tmp_path):
    report = write_report(tmp_path, acceptance_cases())
    assert CHECKER["validate_results"](report, licensed=True, mnist=True) == 11


@pytest.mark.parametrize("change", ["missing", "skipped", "failure", "error", "duplicate"])
@pytest.mark.parametrize("group", ["LICENSED_CASES", "MNIST_CASES"])
def test_ci_rejects_incomplete_acceptance(tmp_path, group, change):
    cases = acceptance_cases()
    target = sorted(CHECKER[group])[0]
    cases = [case for case in cases if case[:2] != target]
    if change == "duplicate":
        cases.extend([(*target, None), (*target, None)])
    elif change != "missing":
        cases.append((*target, change))
    with pytest.raises(ValueError):
        CHECKER["validate_results"](write_report(tmp_path, cases), licensed=True, mnist=True)


def test_ci_allows_delegated_license_cases_in_unlicensed_run(tmp_path):
    report = write_report(tmp_path, [case for case in acceptance_cases()
                                   if case[:2] not in CHECKER["LICENSED_CASES"]])
    assert CHECKER["validate_results"](report, mnist=True) == 9


def test_ci_rejects_empty_report(tmp_path):
    with pytest.raises(ValueError, match="no test cases"):
        CHECKER["validate_results"](write_report(tmp_path, []))


def test_ci_rejects_unrelated_failure(tmp_path):
    report = write_report(tmp_path, acceptance_cases() + [("tests.other", "test_other", "failure")])
    with pytest.raises(ValueError, match="failed tests"):
        CHECKER["validate_results"](report, licensed=True, mnist=True)

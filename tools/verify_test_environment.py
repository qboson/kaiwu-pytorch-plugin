"""Verify the plugin checkout on a CPU, CUDA, or native statistics SDK environment."""

import argparse
from hashlib import sha256
from importlib import import_module, metadata
import json
from pathlib import Path
import platform
import sys
from xml.etree import ElementTree

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    'cuda': 'tests/test_bm_device_sync.py::TestDeviceSyncAcrossAccelerators',
    'sdk': 'tests/test_usage_stats.py::TestNativeCallerContext',
}


def load_checkout():
    """Resolve the plugin from this checkout even with a regular SDK package."""
    source = ROOT / 'src/kaiwu'
    sys.path.insert(0, str(source.parent))
    kaiwu = import_module('kaiwu')
    kaiwu.__path__ = [str(source), *kaiwu.__path__]
    plugin = import_module('kaiwu.torch_plugin')
    if Path(plugin.__file__).resolve() != source / 'torch_plugin/__init__.py':
        raise RuntimeError(f'Unexpected plugin import: {plugin.__file__}')
    return kaiwu, plugin


def preflight(mode, plugin):
    """Exercise real CUDA or confirm native SDK context identity."""
    if mode == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('PyTorch cannot access CUDA in this environment')
        vector = torch.tensor([1.0, 2.0], device='cuda')
        value = (vector @ vector).item()
        torch.cuda.synchronize()
        if value != 5:
            raise RuntimeError('CUDA tensor operation returned an unexpected result')
    elif mode == 'sdk':
        native = import_module('kaiwu.license._track_data')
        usage = import_module(f'{plugin.__name__}.usage_stats')
        context = getattr(native, '_caller_context', None)
        if context is None or context is not usage._kpp_caller_context:
            raise RuntimeError('Installed Kaiwu SDK does not expose a shared statistics caller context')


def main():
    """Run a selected check and record its environment and test results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cpu', 'cuda', 'sdk'))
    parser.add_argument('--output-dir', type=Path, required=True, help='new evidence directory')
    parser.add_argument('--sdk-wheel', type=Path, help='original official SDK wheel for provenance')
    args = parser.parse_args()
    if args.mode == 'sdk' and not args.sdk_wheel:
        parser.error('--sdk-wheel is required for native SDK provenance')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    kaiwu, plugin = load_checkout()
    try:
        sdk_version = metadata.version('kaiwu')
    except metadata.PackageNotFoundError:
        sdk_version = None
    evidence = {
        'mode': args.mode,
        'python': platform.python_version(),
        'interpreter': sys.executable,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'kaiwu': sdk_version,
        'kaiwu_path': getattr(kaiwu, '__file__', None),
        'plugin_path': plugin.__file__,
        'sha256': {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
                   for directory in (ROOT / 'src/kaiwu/torch_plugin', ROOT / 'tests')
                   for path in sorted(directory.rglob('*.py'))},
        'command': [sys.executable, *sys.argv],
    }
    if args.sdk_wheel:
        wheel = args.sdk_wheel.resolve()
        evidence['sdk_wheel'] = {'name': wheel.name, 'sha256': sha256(wheel.read_bytes()).hexdigest()}
    report = output / 'environment.json'
    try:
        if args.mode == 'sdk' and sdk_version is None:
            raise RuntimeError('Native statistics verification requires an installed Kaiwu SDK')
        preflight(args.mode, plugin)
    except (ImportError, RuntimeError) as error:
        evidence['preflight_error'] = str(error)
        report.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
        print(f'Preflight failed: {error}', file=sys.stderr)
        return 1

    target = ROOT / TARGETS.get(args.mode, 'tests')
    xml = output / 'tests.xml'
    options = [str(target), '-q', '-rs', '-p', 'no:cacheprovider',
               f'--basetemp={output / "tmp"}', f'--junitxml={xml}']
    evidence['pytest_args'] = options
    code = pytest.main(options)
    evidence['pytest_exit_code'] = int(code)
    if xml.exists():
        suites = ElementTree.parse(xml).getroot().iter('testsuite')
        totals = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.get(key, '0'))
        evidence['tests'] = totals
        if args.mode in TARGETS and totals != {'tests': 2, 'failures': 0, 'errors': 0, 'skipped': 0}:
            print(f'Acceptance requires both target tests to pass: {totals}', file=sys.stderr)
            code = 1
    elif not code:
        code = 1
    evidence['exit_code'] = int(code)
    report.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    return int(code)


if __name__ == '__main__':
    sys.exit(main())

"""Synchronous CIM factory contracts, with no remote job submission."""
import ast
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import kaiwu
import numpy as np
import pytest
import torch
from kaiwu.torch_plugin import EnergyModel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dplm_sampler', ROOT / 'example/qdiffusion/dplm/models/sampler.py')
sampler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_module)


class Transport:
    def __init__(self, **kwargs):
        self.wait = kwargs['wait']

    def solve(self, matrix):
        return np.ones((1, len(matrix)), dtype=np.float32) if self.wait else None


@pytest.mark.parametrize('explicit', [False, True])
def test_factory_returns_completed_solutions(tmp_path, explicit):
    cim = ModuleType('kaiwu.cim')
    cim.CIMOptimizer = Mock(side_effect=Transport)
    kwargs = {'tmp_dir': str(tmp_path)}
    if explicit:
        kwargs['wait'] = True
    with patch.dict('sys.modules', {'kaiwu.cim': cim}), patch.object(
        kaiwu.common.CheckpointManager, 'save_dir', None
    ):
        sampler = sampler_module.build_bm_sampler(sampler_type='cim', sampler_kwargs=kwargs)
        model = EnergyModel(2, 1, sampler=sampler)
        score = model.score_visible_logits(torch.zeros(2, 2))
    assert cim.CIMOptimizer.call_args.kwargs['wait'] is True
    assert score.shape == (2, 1)
    assert torch.isfinite(score).all()
    assert kwargs == ({'tmp_dir': str(tmp_path), 'wait': True} if explicit else {'tmp_dir': str(tmp_path)})


def test_async_rejected_before_construction_or_directory_changes(tmp_path):
    cim = ModuleType('kaiwu.cim')
    cim.CIMOptimizer = Mock()
    with patch.dict('sys.modules', {'kaiwu.cim': cim}), patch.object(
        kaiwu.common.CheckpointManager, 'save_dir', None
    ), pytest.raises(ValueError, match='wait=True'):
        try:
            sampler_module.build_bm_sampler(sampler_type='cim', sampler_kwargs={
                'wait': False, 'tmp_dir': str(tmp_path)})
        finally:
            cim.CIMOptimizer.assert_not_called()
            assert kaiwu.common.CheckpointManager.save_dir is None


def test_cim_eval_config_default_and_rejection():
    path = ROOT / 'example/qdiffusion/dplm/workflows/esm2_eval.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_cim_eval_config')
    namespace = {'Path': Path, 'EvalConfig': SimpleNamespace}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    kwargs = dict(reference_fasta=Path('reference.fa'), proposal_ckpt='local', energy_ckpt='local',
                  guided_checkpoint='local.pt', output_dir=Path('out'), device='cpu', task_name='test')
    config = namespace['build_cim_eval_config'](**kwargs)
    assert config.bm_sampler_kwargs['wait'] is True
    with pytest.raises(ValueError, match='wait=True'):
        namespace['build_cim_eval_config'](**kwargs, wait=False)

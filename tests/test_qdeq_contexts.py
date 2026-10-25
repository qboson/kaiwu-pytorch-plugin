"""Contract test for the QDEQ lifecycle contexts."""

from __future__ import annotations

import contextlib

from kaiwu.torch_plugin.qdeq import QDEQ


class _Dummy:
    pass


def test_solve_context_is_a_no_op_context_manager():
    context = QDEQ.solve_context(_Dummy())

    assert isinstance(context, contextlib.AbstractContextManager)
    with context as value:
        assert value is None


def test_reattach_context_is_a_no_op_context_manager():
    context = QDEQ.reattach_context(_Dummy())

    assert isinstance(context, contextlib.AbstractContextManager)
    with context as value:
        assert value is None

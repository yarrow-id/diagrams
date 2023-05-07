import pytest
from yarrow.decompose.frobenius import frobenius_decomposition

from hypothesis import given
import hypothesis.strategies as st

from tests.strategies import *
from tests.util import monotonic, is_segmented_arange

@given(c=diagrams())
def test_sort_x_p(c):
    pass

@given(c=diagrams())
def test_frobenius_decomposition(c):
    # Check the function doesn't crash!
    frobenius_decomposition(c)

@given(c=diagrams())
def test_frobenius_decomposition_type(c):
    # Check that a frobenius decomposition has the same type as the diagram being decomposed
    d = frobenius_decomposition(c)
    A1, B1 = c.type
    A2, B2 = d.type
    assert A1 == A2
    assert B1 == B2

@given(c=diagrams())
def test_frobenius_decomposition_monotonic(c):
    # Check that components of a Frobenius decomposition are in
    # (generator, port) order.
    d = frobenius_decomposition(c)

    # (generator, port) order means xi and xo should be arrays of the form
    # 0 0 0 ... | 1 1 1 ... | 2 2 2 ... |
    assert monotonic(d.G.xi.table)
    assert monotonic(d.G.xo.table)

    # similarly, ports should be increasing "runs" (i.e., segmented aranges) of
    # the form
    # 0 1 2 ... | 0 1 2 ... | ...
    assert is_segmented_arange(d.G.pi.table)
    assert is_segmented_arange(d.G.po.table)

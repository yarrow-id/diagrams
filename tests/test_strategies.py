""" Tests for the test strategies! Test all the things! """
from hypothesis import given
import hypothesis.strategies as st
from tests.strategies import *

@given(f=finite_functions(source=2))
def test_healthcheck_finite_functions(f):
    assert (f.target != 0 if f.source != 0 else True)

@given(AB=arrow_type(target=0))
def test_arrow_type_target_zero(AB):
    A, B = AB
    assert B == 0
    assert A == 0

@given(f=finite_functions(target=0))
def test_finite_functions_target_zero(f):
    assert f.source == 0

@given(fg=composite_diagrams())
def test_composite_diagrams(fg):
    f, g = fg
    assert f.type[1] == g.type[0]

@given(sff=segmented_finite_functions())
def test_segmented_finite_function(sff):
    assert sff.sources.source == sff.targets.source
    assert np.sum(sff.sources.table) == sff.values.source
    # values are in the range [0, N)
    # targets are all exactly N, so targets has codomain N+1.
    assert sff.targets.target == sff.values.target +1

@given(ops=operations())
def test_operations(ops):
    xn, s_type, t_type = ops.xn, ops.s_type, ops.t_type
    N = xn.source
    assert len(s_type.sources) == N
    assert len(s_type.targets) == N
    assert len(t_type.sources) == N
    assert len(t_type.targets) == N

    assert np.sum(s_type.sources.table) == s_type.values.source
    assert np.sum(t_type.sources.table) == t_type.values.source

@given(sff_f_wn=object_map_and_half_spider())
def test_object_map_and_half_spider(sff_f_wn):
    sff, f, wn = sff_f_wn
    assert sff.sources.source == wn.target
    assert f.target == wn.source

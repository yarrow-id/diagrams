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

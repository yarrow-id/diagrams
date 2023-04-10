import pytest

import numpy as np
from yarrow.finite_function import FiniteFunction
from yarrow.bipartite_multigraph import BipartiteMultigraph, universal
from yarrow.diagram import Diagram

from hypothesis import given
import hypothesis.strategies as st
from tests.strategies import *

def test_empty():
    # Should run without errors
    e = Diagram.empty()
    (A, B) = e.type
    assert A == FiniteFunction.initial(0)
    assert B == FiniteFunction.initial(0)

@given(wn=finite_functions(), xn=finite_functions(source=0))
def test_identity(wn: FiniteFunction, xn: FiniteFunction):
    # Should run without errors
    d = Diagram.identity(wn, xn)
    (A, B) = d.type
    assert A == wn
    assert B == wn


import pytest

import numpy as np
from yarrow.finite_function import FiniteFunction
from yarrow.bipartite_multigraph import NumpyBipartiteMultigraph, BipartiteMultigraph, universal

from hypothesis import given
import hypothesis.strategies as st
from tests.strategies import *

################################################################################
# Applying coequalizers to bipartite multigraphs

@given(fg=parallel_arrows())
def test_universal_identity(fg):
    f, g = fg
    q = f.coequalizer(g)
    u = universal(q, q)
    assert u == FiniteFunction.identity(q.target)

# A custom strategy for the test_universal_permutation test.
@st.composite
def coequalizer_and_permutation(draw, source=None, target=None):
    f, g = draw(parallel_arrows(source, target))
    q = f.coequalizer(g)
    p = draw(permutations(n=q.target))
    return f, g, q, p

@given(fgqp=coequalizer_and_permutation())
def test_universal_permutation(fgqp):
    f, g, q1, p = fgqp
    q2 = q1 >> p # a permuted coequalizer still coequalizes!
    u = universal(q1, q2)
    assert q1 >> u == q2

################################################################################
# Discrete BPMGs

def test_empty():
    # constructor should run with no errors.
    NumpyBipartiteMultigraph.empty()

@given(wn=finite_functions(), xn=finite_functions(source=0))
def test_discrete(wn, xn):
    # constructor should run with no errors.
    NumpyBipartiteMultigraph.discrete(wn, xn)

@given(f=finite_functions(source=2))
def test_healthcheck_finite_functions(f):
    assert (f.target != 0 if f.source != 0 else True)

# Custom strategy for test_discrete_coequalize_wires
@st.composite
def coequalizer_and_permutation(draw, source=None, target=None):
    f, g = draw(parallel_arrows(source, target))
    W = f.target
    wn   = draw(finite_functions(source=W))
    print(f'target: {W}')
    print(f'wn    : {wn}')
    xn   = draw(finite_functions(source=0))
    return f, g, wn, xn

# Given:
#   f, g : A → W
#   wn : W → Z₁
#   xn : 0 → Z₂
#   D: discrete(wn, xn)
#   q = FF.coequalizer(f, g) : wn.source → Q
@given(cap=coequalizer_and_permutation())
def test_discrete_coequalize_wires(cap):
    f, g, wn, xn = cap
    q = f.coequalizer(g)
    D = NumpyBipartiteMultigraph.discrete(wn, xn)
    E = D.coequalize_wires(q)
    assert E.wn.source == q.target

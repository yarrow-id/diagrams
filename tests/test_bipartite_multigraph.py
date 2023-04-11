import pytest

import numpy as np
from yarrow.finite_function import FiniteFunction
from yarrow.bipartite_multigraph import BipartiteMultigraph, universal

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
    BipartiteMultigraph.empty()

@given(wn=finite_functions(), xn=finite_functions(source=0))
def test_discrete(wn, xn):
    # constructor should run with no errors.
    BipartiteMultigraph.discrete(wn, xn)

# Custom strategy for test_discrete_coequalize_wires
@st.composite
def coequalizer_and_permutation(draw, source=None, target=None, ob=None):
    f, g = draw(parallel_arrows(source, target))
    W = f.target
    wn   = draw(finite_functions(source=W, target=ob))
    print(f'target: {W}')
    print(f'wn    : {wn}')
    xn   = draw(finite_functions(source=0))
    return f, g, wn, xn

# Given:
#   f, g : A → W
#   wn : W → 1
#   xn : 0 → Σ₁
#   D: discrete(wn, xn)
#   q = FF.coequalizer(f, g) : wn.source → Q
# Ensure that wires can be coequalized.
# NOTE: This only handles the PROP case when Σ₀ = 1
@given(cap=coequalizer_and_permutation(ob=1))
def test_discrete_coequalize_unityped_wires(cap):
    f, g, wn, xn = cap
    q = f.coequalizer(g)
    D = BipartiteMultigraph.discrete(wn, xn)
    E = D.coequalize_wires(q)
    assert E.wn.source == q.target

# TODO: FIXME: Need to test coequalize_wires with genuine label-preserving maps!

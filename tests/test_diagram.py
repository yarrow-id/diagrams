import pytest
import unittest

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

@given(ab=parallel_arrows(), xn=finite_functions(source=0))
def test_twist(ab, xn: FiniteFunction):
    wn_A, wn_B = ab

    d = Diagram.twist(wn_A, wn_B, xn)
    (S, T) = d.type
    assert S == wn_A + wn_B
    assert T == wn_B + wn_A

# TODO!
# @unittest.skip
# @given(ab=parallel_arrows(), xn=finite_functions(source=0))
# def test_twist(ab, xn: FiniteFunction):
    # wn_A, wn_B = ab

    # d = Diagram.twist(wn, xn)
    # # TODO: can't implement this as below; no guarantee we'll not get something like (p,
    # # p^{-1}, G) instead!
    # d >> d == Diagram.identity(d.wires)
    # (S, T) = d.type
    # assert S == wn_A + wn_B
    # assert T == wn_B + wn_A


@given(stw=labeled_cospans(), x=finite_functions(source=0))
def test_spider(stw, x):
    """ Given a random cospan
          s   t
        A → W ← B
    And a labeling
        w : W → Σ₀
    Generates a random spider.
    """
    s, t, w = stw
    Diagram.spider(s, t, w, x)

@given(d=spiders())
def test_dagger_spider(d: Diagram):
    e = d.dagger()
    assert e.s == d.t
    assert e.t == d.s

    X = e.type
    Y = d.type
    assert X[0] == Y[1]
    assert X[1] == Y[0]

    assert e.G == d.G

# from hypothesis import settings, reproduce_failure
# @settings(print_blob=True)
@given(abx=generator_and_typing())
def test_singleton(abx):
    a, b, xn = abx
    # test constructor works
    d = Diagram.singleton(a, b, xn)
    (A, B) = d.type
    # Type of the diagram should be the same as the chosen typing of the generator.
    assert a == A
    assert b == B
    # The number of internal wires should be equal to the number of ports on the
    # generator.
    assert d.wires == a.source + b.source

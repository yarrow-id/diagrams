import numpy as np
import unittest
from hypothesis import given
from tests.strategies import objects, adapted_function, finite_functions, permutations, parallel_permutations, parallel_arrows

from yarrow.numpy import FiniteFunction
from yarrow.finite_function import argsort

from tests.util import sorts

# Invert a permutation
def invert(p):
    return argsort(p)

# Ensure the invert function works(!)
@given(p=permutations())
def test_invert(p):
    assert invert(p) >> p == FiniteFunction.identity(p.source)
    assert p >> invert(p) == FiniteFunction.identity(p.source)

# Definition A.2 "Sorting"
@given(f=finite_functions())
def test_argsort_matches_definition(f):
    p = f.argsort()
    y = p >> f

    if len(y.table) <= 1:
        return None

    assert sorts(p, f)

# Proposition A.3
# we test something slightly weaker; instead of a general monomorphism we just
# use a permutation.
# TODO: generate a monomorphism by just `spreading out' values of the identity
# function, then permuting?
@given(p=permutations())
def test_argsort_monomorphism_strictly_increasing(p):
    q = p.argsort()
    y = q >> p

    if len(y.table) <= 1:
        return None

    assert sorts(q, p, strict=True)

# TODO: test uniqueness A.4 (?)

# Proposition A.5
@given(fpq=adapted_function(source=None, target=None))
def test_sort_by_permuted_key(fpq):
    f, p, q = fpq
    s = f.argsort()
    assert sorts(s >> invert(p), p >> f)

# Proposition A.6
# Again using permutations instead of monomorphisms;
# see test_argsort_monomorphism_strictly_increasing
@given(fp=parallel_permutations())
def test_sort_pf_equals_sortf_p(fp):
    f, p = fp
    assert (p >> f).argsort() == (f.argsort() >> invert(p))

# interleave and its inverse cancel on both sides
@given(n=objects)
def test_interleave_inverse(n: int):
    a = FiniteFunction.interleave(n)
    b = FiniteFunction.cointerleave(n)
    i = FiniteFunction.identity(2*n)

    assert a >> b == i
    assert b >> a == i

# Cointerleaving is the opposite of interleaving, and has a more meaningful
# interpretation which we can test easily.
@given(fg=parallel_arrows())
def test_cointerleave(fg):
    f, g = fg
    N = f.source
    assert N == g.source # should be true because parallel_arrows

    h = (f @ g)
    a = FiniteFunction.cointerleave(N)
    r = a >> h

    Array = type(f)._Array

    assert Array.all(r.table[0::2] == h.table[0:N])
    assert Array.all(r.table[1::2] == h.table[N:])

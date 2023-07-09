import pytest

import numpy as np
from yarrow.finite_function import FiniteFunction, bincount, cumsum

from hypothesis import given
import hypothesis.strategies as st

from tests.strategies import *

################################################################################
# Equality

@given(f=finite_functions())
def test_equality_reflexive(f):
    assert f == f

@given(fg=parallel_arrows())
def test_inequality_table(fg):
    """ Ensure that if the function tables of two FiniteFunctions are different,
    then == returns false."""
    f, g = fg
    if np.any(f.table != g.table):
        assert f != g

@given(f=finite_functions(), g=finite_functions())
def test_inequality_type(f, g):
    """ Ensure that if the function tables of two FiniteFunctions are different,
    then == returns false."""
    if f.source != g.source:
        assert f != g
    if f.target != g.target:
        assert f != g

################################################################################
# Basic tests

# only test small arrays, we don't need to OOM thank you very much
@given(objects)
def test_identity(n):
    identity = FiniteFunction.identity(n)
    assert np.all(identity.table == np.arange(0, n))
    assert identity.table.shape == (n, )

@given(st.integers(max_value=-1))
def test_identity_invalid_object(n):
    with pytest.raises(AssertionError) as exc_info:
        identity = FiniteFunction.identity(n)


################################################################################
# Category laws

# left identities       id ; f = f
@given(f=finite_functions(source=0))
def test_identity_composition_left(f):
    """ id ; f = f """
    identity = FiniteFunction.identity(f.source)
    assert (identity >> f) == f

# right identities      f ; id = f
@given(f=finite_functions())
def test_identity_composition_right(f):
    """ f ; id = f """
    identity = FiniteFunction.identity(f.target)
    assert (f >> identity) == f

# Make sure composition doesn't crash
@given(fns=composite_functions(n=2))
def test_composition(fns):
    f, g = fns
    x = f >> g

# Check associativity of composition    (f ; g) ; h = f ; (g ; h)
@given(fgh=composite_functions(n=3))
def test_composition_assoc(fgh):
    f, g, h = fgh
    assert ((f >> g) >> h) == (f >> (g >> h))

################################################################################
# Coproducts

# Uniqueness of the initial map
# any map f : 0 → B is equal to the initial map ? : 0 → B
@given(f=finite_functions(source=0))
def test_initial_map_unique(f):
    assert f == FiniteFunction.initial(f.target)

# Coproducts!
# given f : A₁ → B and g : A₂ → B, ensure f + g commutes with injections.
# i.e.,  ι₀ ; (f + g) = f
#        ι₁ ; (f + g) = g
@given(fg=composite_coproduct())
def test_coproduct_commutes(fg):
    f, g = fg
    i0 = FiniteFunction.inj0(f.source, g.source)
    i1 = FiniteFunction.inj1(f.source, g.source)

    assert (i0 >> (f + g)) == f
    assert (i1 >> (f + g)) == g


@given(f=finite_functions(), b=objects)
def test_f_cp_inj0_equals_inject0(f, b):
    assert f >> FiniteFunction.inj0(f.target, b) == f.inject0(b)

@given(f=finite_functions(), a=objects)
def test_f_cp_inj1_equals_inject0(f, a):
    assert f >> FiniteFunction.inj1(a, f.target) == f.inject1(a)

################################################################################
# (Strict) symmetric monoidal tests

@given(f=finite_functions(), g=finite_functions())
def test_tensor_vs_injections(f, g):
    """ Verify that the tensor product corresponds to its definition in terms of
    coproducts and injections """
    i0 = FiniteFunction.inj0(f.target, g.target)
    i1 = FiniteFunction.inj1(f.target, g.target)

    f @ g == (f >> i0) + (g >> i1)

@given(a=objects, b=objects)
def test_twist_inverse(a, b):
    """ Check the law σ ; σ = id """
    f = FiniteFunction.twist(a, b)
    g = FiniteFunction.twist(b, a)
    identity = FiniteFunction.identity(a + b)
    assert f >> g == identity
    assert g >> f == identity

@given(f=finite_functions(), g=finite_functions())
def test_twist_naturality(f, g):
    """ Check naturality of σ, so that (f @ g) ; σ = σ ; (f @ g) """
    post_twist = FiniteFunction.twist(f.target, g.target)
    pre_twist  = FiniteFunction.twist(f.source, g.source)
    assert ((f @ g) >> post_twist) == (pre_twist >> (g @ f))

################################################################################
# Test coequalizers
@given(fg=parallel_arrows())
def test_coequalizer_commutes(fg):
    f, g = fg
    c = f.coequalizer(g)
    assert (f >> c) == (g >> c)

################################################################################
# Finite coproducts

@given(s=finite_functions())
def test_injection_coproduct_identity(s: FiniteFunction):
    """ Test that
        ι_0 + ι_1 + ... + ι_N = identity(sum_{i ∈ N} s(i))
    """
    i = FiniteFunction.identity(s.source)
    assert s.injections(i) == FiniteFunction.identity(np.sum(s.table))

@given(f=finite_functions())
def test_bincount(f: FiniteFunction):
    b = bincount(f)
    assert b.source == f.target
    assert b.target == f.source+1
    assert np.all(b.table < len(f)+1)
    assert np.all(b.table >= 0)

################################################################################
# Test functions with None target

@given(fg=composite_coproduct_finite_domain())
def test_finite_domain_coproduct(fg):
    f, g = fg
    A0, B = f.type
    A1, B_ = g.type
    assert B == B_

    h = f + g
    X, Y = h.type

    assert X == A0 + A1
    assert Y == B
    assert h.table.dtype == f.table.dtype
    assert h.table.dtype == g.table.dtype

@given(fg=composite_nonfinite_codomain())
def test_compose_finite_domain(fg):
    """ Test composition with functions of non-finite codomain """
    f, g = fg

    A, B = f.type
    B_, C = g.type
    assert B == B_

    h = f >> g
    assert h.type == (A, C)

@given(f=finite_functions())
def test_cumsum(f):
    g = cumsum(f)
    assert len(g) == len(f)
    assert g.target == np.sum(f.table) + 1

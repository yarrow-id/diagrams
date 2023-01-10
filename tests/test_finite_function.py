import pytest

import numpy as np
from yarrow.finite_function import FiniteFunction

from hypothesis import given
import hypothesis.strategies as st

from tests.strategies import *

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

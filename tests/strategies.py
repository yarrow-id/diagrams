""" Hypothesis strategies for FiniteFunctions """
import numpy as np
from yarrow.finite_function import FiniteFunction
import hypothesis.strategies as st

# a generator for objects of FinFun
objects = st.integers(min_value=0, max_value=32)

def _is_valid_arrow_type(s, t):
    if t == 0:
        return s == 0
    return True

# Generate a pair of objects corresponding to the source/target of a morphism.
@st.composite
def arrow_type(draw, source=None, target=None):
    if target == 0 and source != 0:
        raise ValueError("No arrows exist of type n → 0 for n != 0.")

    if target == None:
        target = draw(objects)

    if target == 0:
        source = 0
    elif source == None:
        source = draw(objects)

    return source, target


# generate a random FiniteFunction
@st.composite
def finite_functions(draw, source=None, target=None):
    source, target = draw(arrow_type(source=source, target=target))
    assert _is_valid_arrow_type(source, target)

    if target == 0:
        return FiniteFunction(target, np.zeros(0, dtype=int))

    # generate a random array of elements in {0, ..., target - 1}
    assert target > 0
    elements = st.integers(min_value=0, max_value=target-1)
    table = draw(st.lists(elements, min_size=source, max_size=source))
    return FiniteFunction(target, table)

# Generate exactly n composite functions.
@st.composite
def composite_functions(draw, n):
    # NOTE: n is the number of *arrows*, so we need n+1 *objects*.
    #
    #    f₁   f₂ ... fn
    # A₀ → A₁ →  ... → An

    if n == 0:
        return []

    # for each function f : A → B, if B = 0, then A = 0.
    # This is because there can be no functions n → 0 with n != 0
    obj = draw(st.lists(objects, min_size=n+1, max_size=n+1))
    for i in range(0, n+1):
        if obj[-i] == 0:
            obj[-i-1] = 0

    fs = [ draw(finite_functions(source=a, target=b)) for a, b in zip(obj, obj[1:]) ]
    return fs

# generate h : X → B,
# and from this generate f : A₁ → B, g : A₂ → B
# such that A₁ + A₂ = X
# and f + g = h
@st.composite
def composite_coproduct(draw, source=None, target=None):
    source, target = draw(arrow_type(source, target))
    assert _is_valid_arrow_type(source, target)
    a1 = draw(st.integers(min_value=0, max_value=source))
    a2 = source - a1

    f = draw(finite_functions(source=a1, target=target))
    g = draw(finite_functions(source=a2, target=target))

    return f, g

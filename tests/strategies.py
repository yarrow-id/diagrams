""" Hypothesis strategies for FiniteFunctions """
import numpy as np
from yarrow.finite_function import FiniteFunction
import hypothesis.strategies as st

# a generator for objects of FinFun
objects = st.integers(min_value=0, max_value=32)

# generate a random FiniteFunction
@st.composite
def finite_functions(draw, source=None, target=None):
    if target == 0 and source != 0:
        raise ValueError("Can't generate a FiniteFunction of type n → 0 for n != 0.")

    if target == None:
        target = draw(objects)

    if target == 0:
        source = 0
    elif source == None:
        source = draw(objects)

    # generate a random array of elements in {0, ..., target - 1}
    if target == 0:
        return FiniteFunction(target, np.zeros(0, dtype=int))

    assert target > 0
    elements = st.integers(min_value=0, max_value=target-1)
    table = draw(st.lists(elements, min_size=source, max_size=source))
    return FiniteFunction(target, table)

# Generate exactly n composite functions.
@st.composite
def composite_functions(draw, n):
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

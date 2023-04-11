""" Hypothesis strategies for FiniteFunctions """
import numpy as np
from yarrow.finite_function import FiniteFunction
from yarrow.diagram import Diagram
import hypothesis.strategies as st

# a generator for objects of FinFun
objects = st.integers(min_value=0, max_value=32)
nonzero_objects = st.integers(min_value=1, max_value=32)

def _is_valid_arrow_type(s, t):
    if t == 0:
        return s == 0
    return True

@st.composite
def arrow_type(draw, source=None, target=None):
    """ Generate a random type of finite function.
    For example, a type of n → 0 is forbidden.
    """
    # User specified both target and source
    if target is not None and source is not None:
        if target == 0 and source != 0:
            raise ValueError("No arrows exist of type n → 0 for n != 0.")
        return source, target

    elif source is None:
        # any target
        target = draw(objects) if target is None else target

        if target == 0:
            source = 0
        else:
            source = draw(objects)

        return source, target

    # target is None, but source is not
    target = draw(nonzero_objects) if source > 0 else draw(objects)
    return source, target

# generate a random FiniteFunction
@st.composite
def finite_functions(draw, source=None, target=None):
    source, target = draw(arrow_type(source=source, target=target))
    assert _is_valid_arrow_type(source, target)

    # generate a random array of elements in {0, ..., target - 1}
    if target == 0:
        # FIXME: remove np hardcoding for other backends.
        table = np.zeros(0, dtype=int)
    else:
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

@st.composite
def parallel_arrows(draw, source=None, target=None):
    source, target = draw(arrow_type(source, target))
    assert _is_valid_arrow_type(source, target)

    f = draw(finite_functions(source=source, target=target))
    g = draw(finite_functions(source=source, target=target))
    return f, g

@st.composite
def parallel_permutations(draw, source=None, target=None):
    n = draw(objects)
    assert _is_valid_arrow_type(n, n)
    p = draw(permutations(n))
    q = draw(permutations(n))
    return p, q

@st.composite
def permutations(draw, n=None):
    if n is None:
        n = draw(objects)
    x = np.arange(0, n, dtype=int)
    np.random.shuffle(x)
    return FiniteFunction(n, x)

@st.composite
def adapted_function(draw, source=None, target=None):
    source, target = draw(arrow_type(source, target))
    assert _is_valid_arrow_type(source, target)

    f = draw(finite_functions(source=source, target=target))
    p = draw(permutations(n=source))
    q = draw(permutations(n=target))

    return f, p, q

# Draw a cospan
#   s : A → W
#   t : B → W
#   w : W → Σ₀
@st.composite
def labeled_cospans(draw, W=None, Ob=None, A=None, B=None):
    w = draw(finite_functions(source=W, target=Ob))
    s = draw(finite_functions(source=A, target=w.source))
    t = draw(finite_functions(source=B, target=w.source))
    return (s, t, w)

@st.composite
def spiders(draw, W=None, Ob=None, A=None, B=None, Arr=None):
    s, t, w = draw(labeled_cospans(W=W, Ob=Ob, A=A, B=B))
    x = draw(finite_functions(source=0, target=Arr))
    return Diagram.spider(s, t, w, x)

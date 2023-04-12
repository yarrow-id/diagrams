""" Hypothesis strategies for FiniteFunctions """
import numpy as np
from yarrow.array import numpy
from yarrow.finite_function import FiniteFunction
from yarrow.bipartite_multigraph import BipartiteMultigraph
from yarrow.diagram import Diagram
import hypothesis.strategies as st

_MAX_GENERATORS = 32
_MAX_OBJECTS = 32

# generator for finite sets Σ₁
generators = st.integers(min_value=0, max_value=_MAX_GENERATORS)

# a generator for objects of FinFun
objects = st.integers(min_value=0, max_value=_MAX_OBJECTS)
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
        table = np.random.randint(0, high=target, size=source)

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

################################################################################
# Diagrams

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
    """ Given a random cospan
          s   t
        A → W ← B
    And a labeling
        w : W → Σ₀
    Generate a random spider.
    """
    s, t, w = draw(labeled_cospans(W=W, Ob=Ob, A=A, B=B))

    x = draw(finite_functions(source=0, target=Arr))
    return Diagram.spider(s, t, w, x)

@st.composite
def generator_and_typing(draw):
    """ Generate a random generator
        x : 1 → Σ₁
    and its type
        a : A → Σ₀
        b : B → Σ₀
    """
    # Σ₁ > 0, Σ₀ > 0
    Arr = draw(nonzero_objects)
    Obj = draw(nonzero_objects)

    # xn : 1 → Σ₁
    xn = draw(finite_functions(source=1, target=Arr))

    # Typing
    a = draw(finite_functions(target=Obj))
    b = draw(finite_functions(target=Obj))

    return a, b, xn

@st.composite
def singletons(draw):
    a, b, xn = draw(generator_and_typing())
    return Diagram.singleton(a, b, xn)

@st.composite
def diagrams(draw, Obj=None, Arr=None):
    """ Generate a random diagram.
    Since we're also generating a random signature,
    we only need to ensure that ports are correct.
    """
    Array = numpy

    # Σ₀ > 0    Σ₁ ≥ 0
    Obj = Obj if Obj is not None else draw(nonzero_objects)
    Arr = Arr if Arr is not None else draw(objects)

    # max arity, coarity of generators.
    # These get set to 0 if there are no wires in the diagram.
    MAX_ARITY = None
    MAX_COARITY = None

    # Start with the number of wires in the diagram
    # NOTE: This probably biases generation somehow.
    wn = draw(finite_functions(target=Obj))

    if wn.source == 0 or Arr == 0:
        MAX_ARITY = 0
        MAX_COARITY = 0
        # return Diagram.empty(wn)

    # 'arities' maps each generator xn(i) to its arity
    arities   = draw(finite_functions(target=MAX_ARITY))
    Ei = np.sum(arities.table)

    coarities = draw(finite_functions(source=arities.source, target=MAX_COARITY))
    Eo = np.sum(coarities.table)

    # Now choose the number of generators and their arity/coarity.
    xn = draw(finite_functions(source=arities.source, target=Arr))

    # wi : Ei → W
    # NOTE: Hypothesis builtin strategies really don't like numpy's int64s!
    wi = draw(finite_functions(source=int(Ei), target=wn.source))
    wo = draw(finite_functions(source=int(Eo), target=wn.source))

    # pi and po are a 'segmented arange' of the arities and coarities
    # e.g., [ 3 2 0 5 ] → [ 0 1 2 0 1 0 1 2 3 4 ]
    pi = FiniteFunction(None, Array.segmented_arange(arities.table))
    po = FiniteFunction(None, Array.segmented_arange(coarities.table))

    # relatedly, xi and xo are just a repeat:
    # (TODO: we could inline segmented_arange here and save recomputation of e.g., repeats)
    # e.g., [ 3 2 0 5 ] → [ 0 0 0 1 1 2 2 2 2 2 ]
    i = Array.arange(xn.source, dtype=int)
    xi = FiniteFunction(xn.source, Array.repeat(i, arities.table))
    xo = FiniteFunction(xn.source, Array.repeat(i, coarities.table))

    G = BipartiteMultigraph(
            wi=wi,
            wo=wo,

            xi=xi,
            xo=xo,

            wn=wn,
            pi=pi,
            po=po,
            xn=xn)

    s = draw(finite_functions(target=wn.source))
    t = draw(finite_functions(target=wn.source))
    return Diagram(s, t, G)

@st.composite
def many_diagrams(draw, n):
    """ Generate several diagrams from the same signature """
    # TODO: allow Obj = 0? Then we can only ever generate the empty diagram, or
    # maybe only diagrams with generating morphisms of type 0 → 0?
    Obj = draw(nonzero_objects)
    Arr = draw(generators)
    result = []
    return [ draw(diagrams(Obj=Obj, Arr=Arr)) for _ in range(0, n) ]

@st.composite
def composite_diagrams(draw, max_boundary_size=128):
    """
    Generate a composite diagram with a random signature.
          f ; g
        A → B → C
    """
    # Obj = draw(nonzero_objects)
    Obj = 1 # Only handle the PROP case for now.
    Arr = draw(generators)

    # Draw two diagrams with Σ₀ = 1, then change sources + targets to have a
    # common boundary.
    f = draw(diagrams(Obj=Obj, Arr=Arr))
    g = draw(diagrams(Obj=Obj, Arr=Arr))

    if f.wires == 0 or g.wires == 0:
        B = 0
    else:
        B = draw(st.integers(min_value=0, max_value=max_boundary_size))

    f.t = draw(finite_functions(source=B, target=f.wires))
    g.s = draw(finite_functions(source=B, target=g.wires))

    return f, g

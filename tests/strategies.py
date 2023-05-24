""" Hypothesis strategies for FiniteFunctions """
import numpy as np
from yarrow.array import numpy
from yarrow.finite_function import FiniteFunction
from yarrow.bipartite_multigraph import BipartiteMultigraph
from yarrow.diagram import Diagram

from yarrow.segmented.operations import Operations
from yarrow.segmented.finite_function import SegmentedFiniteFunction

import hypothesis.strategies as st

# these constants are completely arbitrary, I picked a smallish number I like.
_MAX_SEGMENT_SIZE = 32
_MAX_SIGMA_1 = 32
_MAX_OBJECTS = 32

# generator for arity/coarity of operations
segment_sizes = st.integers(min_value=0, max_value=_MAX_SEGMENT_SIZE)

# generator for finite sets Σ₁
sigma_1 = st.integers(min_value=0, max_value=_MAX_SIGMA_1)
nonzero_sigma_1 = st.integers(min_value=1, max_value=_MAX_SIGMA_1)

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

@st.composite
def finite_function_lists(draw, n=None, source=None, target=None):
    """ Draw a small-ish list of N finite functions, and an indexer x : X → N"""
    n = n if n is not None else draw(objects)
    fs = [ draw(finite_functions(source=source, target=target)) ]
    x = draw(finite_functions(target=len(fs)))
    return fs, x

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
def common_targets(draw, n=None, source=None, target=None):
    """ Draw a list of random functions with the same target,

        fs[i] : A_i → B

    For i ∈ N, and an indexing function

        x : X → N
    """
    n = draw(objects) if n is None else n

    target = draw(objects) if target is None else target

    fs = []
    for i in range(0, n):
        source, _ = draw(arrow_type(target=target))
        fs.append(draw(finite_functions(source=source, target=target)))

    x = draw(finite_functions(target=n))
    return fs, x

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
def generator_and_typing(draw, Obj=None, Arr=None):
    """ Generate a random generator
        x : 1 → Σ₁
    and its type
        a : A → Σ₀
        b : B → Σ₀
    """
    # Σ₁ > 0, Σ₀ > 0
    Arr = draw(nonzero_objects) if Arr is None else Arr
    Obj = draw(nonzero_objects) if Obj is None else Obj

    # xn : 1 → Σ₁
    xn = draw(finite_functions(source=1, target=Arr))

    # Typing
    a = draw(finite_functions(target=Obj))
    b = draw(finite_functions(target=Obj))

    return a, b, xn

@st.composite
def singletons(draw, Obj=None, Arr=None):
    a, b, xn = draw(generator_and_typing(Obj=Obj, Arr=Arr))
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
    Arr = draw(sigma_1)
    return [ draw(diagrams(Obj=Obj, Arr=Arr)) for _ in range(0, n) ]

@st.composite
def many_singletons(draw, n):
    """ Generate several singleton diagrams from the same signature """
    Obj = draw(objects)
    Arr = draw(nonzero_sigma_1)
    return [ draw(singletons(Obj=Obj, Arr=Arr)) for _ in range(0, n) ]

@st.composite
def composite_diagrams(draw, max_boundary_size=128):
    """
    Generate a composite diagram with a random signature.
          f ; g
        A → B → C
    """
    # Obj = draw(nonzero_objects)
    Obj = 1 # Only handle the PROP case for now.
    Arr = draw(sigma_1)

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

@st.composite
def composite_singletons(draw, max_boundary_size=128):
    """
    Generate a composite diagram with a random signature.
          f ; g
        A → B → C
    where f, g are singleton diagrams.
    """

    Σ_0 = draw(objects)

    # types of f, g
    a = draw(finite_functions(target=Σ_0))
    b = draw(finite_functions(target=Σ_0))
    c = draw(finite_functions(target=Σ_0))

    # generator labels
    _, Σ_1 = draw(arrow_type(source=1))
    xn_f = draw(finite_functions(source=1, target=Σ_1))
    xn_g = draw(finite_functions(source=1, target=Σ_1))
    f = Diagram.singleton(a, b, xn_f)
    g = Diagram.singleton(b, c, xn_g)

    return f, g

################################################################################
# Segmented finite functions


# Generate a segmented finite function like the one below
#   sff
#       sources: N            → K₀
#       values : sum(sources) → Σ₀      (= max(targets))
#       targets: N            → Σ₀      (= const Σ₀+1)
@st.composite
def segmented_finite_functions(draw, N=None, Obj=None):
    N, Obj = draw(arrow_type(source=N, target=Obj))

    sources = draw(finite_functions(source=N))
    values  = draw(finite_functions(source=np.sum(sources.table), target=Obj))

    # make an array [Σ₀, Σ₀, ... ]
    targets = FiniteFunction.terminal(N).inject1(Obj)

    return SegmentedFiniteFunction(
        sources=sources,
        targets=targets,
        values=values)

# Generate a tensoring of operations with the following types.
#   xn         : N            → Σ₁
#
#   s_type
#       sources: N            → K₀
#       values : sum(sources) → Σ₀      (= max(targets))
#       targets: N            → Σ₀      (= const Σ₀+1)
#   t_type
#       sources: N            → K₁
#       values : sum(sources) → Σ₀      (= max(targets))
#       targets: N            → Σ₀      (= const Σ₀+1)
@st.composite
def operations(draw):
    Obj = draw(objects)
    s_type = draw(segmented_finite_functions(Obj=Obj))
    t_type = draw(segmented_finite_functions(
        N=len(s_type.sources),
        Obj=s_type.values.target))

    N = len(s_type.sources)
    xn = draw(finite_functions(source=N))

    return Operations(xn, s_type, t_type)

@st.composite
def half_spider(draw, Obj=None):
    Obj = draw(objects) if Obj is None else Obj
    wn = draw(finite_functions(target=Obj))
    f  = draw(finite_functions(target=wn.source))
    return f, wn

# The functions
#   f  : A → B
#   wn : B → Σ₀
# together give a half-spider, and
#   F₀~ : Σ₀ → Ω₀*
#   F₀  : sum(s) → Ω₀
#   s   : Σ₀ → Nat
# a segmented array encoding the object map of a (finite) functor.
@st.composite
def object_map_and_half_spider(draw):
    sff = draw(segmented_finite_functions())
    f, wn = draw(half_spider(Obj=sff.sources.source))
    return sff, f, wn


################################################################################
# Functions of Finite Domain
# TODO: these generators are quite hacky and not nice. Refactor!

# A FinFun target is either:
#   (target: int, dtype=int64)
#   (None,        dtype=[int64 | object])
#

@st.composite
def finite_function_target(draw, target=None, is_inf=None, inf_dtype=None):
    if target is not None:
        return False, None

    is_inf = draw(st.booleans()) if is_inf == None else is_inf
    inf_dtype = draw(st.sampled_from(['int', 'object'])) if inf_dtype is None else inf_dtype

    return is_inf, inf_dtype

@st.composite
def finite_domain_functions(draw, source=None, target=None, is_inf=None, inf_dtype=None):
    """ Generate functions of finite domain (possibly infinite codomain!) """
    is_inf, inf_dtype = draw(finite_function_target(target, is_inf, inf_dtype))

    if is_inf:
        f = draw(finite_functions(source=source))
        target = None
        dtype = inf_dtype # if inf_dtype is not None else draw(st.sampled_from(['int', 'object']))
        if dtype == 'int':
            table = f.table
        else:
            # TODO: Generate more varied object data
            table = np.empty(len(f.table), dtype='object')
            table[:] = [(x,x) for x in f.table]
    else:
        f = draw(finite_functions(source=source, target=target))
        target = f.target
        table = f.table

    return FiniteFunction(target, table, dtype=table.dtype)

@st.composite
def composite_coproduct_finite_domain(draw, source=None, target=None):
    source, target = draw(arrow_type(source, target))
    assert _is_valid_arrow_type(source, target)

    a1 = draw(st.integers(min_value=0, max_value=source))
    a2 = source - a1

    is_inf, inf_dtype = draw(finite_function_target(target=target))
    f = draw(finite_domain_functions(source=a1, target=target, is_inf=is_inf, inf_dtype=inf_dtype))
    g = draw(finite_domain_functions(source=a2, target=target, is_inf=is_inf, inf_dtype=inf_dtype))

    return f, g

@st.composite
def composite_nonfinite_codomain(draw, source=None, middle=None):
    """ Draw a composite function with a possibly non-finite codomain """
    A, B = draw(arrow_type(source, middle))

    f = draw(finite_functions(source=A, target=B))
    g = draw(finite_domain_functions(source=B))
    return f, g

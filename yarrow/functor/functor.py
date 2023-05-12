from typing import List
from abc import abstractmethod

from yarrow.diagram import Diagram
from yarrow.finite_function import AbstractFiniteFunction, FiniteFunction, bincount
from yarrow.decompose.frobenius import frobenius_decomposition
from yarrow.segmented.finite_function import AbstractSegmentedFiniteFunction, SegmentedFiniteFunction
from yarrow.segmented.operations import Operations

class Functor:
    """ A base class for implementing strict symmetric monoidal hypergraph functors """
    # In the paper we denote
    # - map_objects as F₀
    # - map_arrow   as F₁
    @abstractmethod
    def map_objects(self, objects: AbstractFiniteFunction) -> AbstractSegmentedFiniteFunction:
        """Given an array of objects encoded as a FiniteFunction

            objects : W → Σ₀

        Return *segmented* finite function representing the lists to which each
        object was mapped.

            sources : W            → Nat
            targets : W            → Nat
            values  : sum(sources) → Ω₀
        """
        ...

    @abstractmethod
    def map_arrow(self, d: Diagram) -> Diagram:
        """F.map_arrow(d) should apply the functor F to diagram d."""
        ...

def apply_finite_object_map(
        finite_object_map: AbstractSegmentedFiniteFunction,
        wn: AbstractFiniteFunction) -> AbstractSegmentedFiniteFunction:
    """Given an AbstractSegmentedFiniteFunction f representing a family of K functions

        f_i : N_i → Ω₀*, i ∈ K

    where f_i is the action of a functor F on generating object i,
    and an object wn = L(A) in the image of L,
    apply_finite_object_map(f, wn) computes the object F(wn)
    as a segmented array.
    """
    return type(finite_object_map)(
        values  = finite_object_map.sources.injections(wn) >> finite_object_map.values,
        sources = wn >> finite_object_map.sources,
        targets = wn >> finite_object_map.targets)

def map_half_spider(swn: AbstractSegmentedFiniteFunction, f: AbstractFiniteFunction):
    """Let swn = F.map_objects(wn) for some functor F,
    and suppose S(f) is a half-spider.
    Then map_half_spider(swn, f) == F(S(f)).
    """
    # NOTE: swn should be the result of applying an object map to wn.
    # swn = object_map(wn)
    return swn.sources.injections(f)

def decomposition_to_operations(d: 'Diagram'):
    """ Get the array of operations (and their types) from a Frobenius
    decomposition.  """
    # NOTE: it's *very* important that d is a frobenius decomposition, since we
    # directly use the maps d.G.wi and d.G.wo in the result.
    Fun = d._Fun
    Array = Fun._Array
    s_type = SegmentedFiniteFunction(
        sources = bincount(d.G.xi),
        targets = Fun(None, Array.full(d.operations, d.G.xn.target)),
        values  = d.G.wi >> d.G.wn)

    t_type = SegmentedFiniteFunction(
        sources = bincount(d.G.xo),
        targets = Fun(None, Array.full(d.operations, d.G.xn.target)),
        values  = d.G.wo >> d.G.wn)
    
    return Operations(d.G.xn, s_type, t_type)

class FrobeniusFunctor(Functor):
    """ A functor defined in terms of Frobenius decompositions.
    Instead of specifying map_arrow, the implementor can specify map_operations,
    which should map a tensoring of generators to a tensoring of diagrams.
    """
    @abstractmethod
    def map_objects(self, objects: AbstractFiniteFunction) -> SegmentedFiniteFunction:
        ...

    def map_arrow(self, d: Diagram):
        d = frobenius_decomposition(d)
        ops = decomposition_to_operations(d)

        swn = self.map_objects(d.G.wn)
        h = self.map_operations(ops)

        Fun   = h._Fun
        Graph = h._Graph

        xn = d._Fun.initial(h.G.xn.target)

        # build the morphisms (s ; x) ; (id × h) ; (z ; t) from Proposition B.1
        i = Diagram.identity(swn.values, xn)
        # note: we use the source/target maps of i in constructing those of sx, yt
        # to avoid constructing another array with the same data.
        sx = Diagram(d.s, i.t + map_half_spider(swn, d.G.wi), Graph.discrete(swn.values, xn))
        yt = Diagram(i.s + map_half_spider(swn, d.G.wo), d.t, Graph.discrete(swn.values, xn))
        return (sx >> (i @ h) >> yt)

    @abstractmethod
    def map_operations(self, ops: Operations) -> Diagram:
        """Given an array of generating operations

            xn : X → Σ₁

        and their types (encoded as segmented arrays)

            s_type : sum_{i ∈ X} arity(xn(i))   → Σ₀
            t_type : sum_{i ∈ X} coarity(xn(i)) → Σ₀

        Return a Diagram representing the tensoring

            F₁(xn(0)) ● F₁(xn(1)) ... F₁(xn(X - 1))
        """
        ...

################################################################################
# Built-in functors, supplied as examples.

def identity_object_map(objects: AbstractFiniteFunction):
    """ The object map of the identity functor """
    Fun = type(objects)
    Array = objects._Array
    return SegmentedFiniteFunction(
        sources = Fun(2, Array.ones(len(objects))),
        targets = Fun(objects.target+1, Array.full(len(objects), objects.target)),
        values  = objects)

class Identity(Functor):
    """ The identity functor, implemented by actually just returning the same
    diagram """
    def map_objects(self, objects):
        return _identity_object_map(objects)
        
    def map_arrow(self, d: Diagram):
        return d

class FrobeniusIdentity(FrobeniusFunctor):
    """ The identity functor, implemented using Frobenius decompositions.
    This is provided as a simple example of how to use the FrobeniusFunctor type:
    instead of implementing map_arrow directly, one can instead write a mapping
    on tensorings of operations.
    This is typically much easier, since for a strict monoidal functor F we have
    F(f₀ ● f₁ ● f₂ ... fn) = F(f₀) ● F(f₁) ● F(f₂) ● ... ● F(fn) 
    """
    def map_objects(self, objects: AbstractFiniteFunction):
        return identity_object_map(objects)

    def map_operations(self, ops: Operations) -> Diagram:
        return Diagram.tensor_operations(ops)

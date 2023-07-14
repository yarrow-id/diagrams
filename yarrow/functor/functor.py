""" Strict Symmetric Monoidal Hypergraph Functors of Diagrams.

This module consists of two classes.

* :py:class:`Functor`
* :py:class:`FrobeniusFunctor`

The former is the generic interface that all functors must implement.
If you wish to define a functor, it is strongly recommended to define
a class which inherits from :py:class:`FrobeniusFunctor`.
That way, instead of mapping arbitrary diagrams, you can define your functor in
terms of mapping *generating operations*.

"""
from abc import abstractmethod
from yarrow.finite_function import AbstractFiniteFunction, bincount
from yarrow.diagram import AbstractDiagram
from yarrow.segmented.finite_function import AbstractIndexedCoproduct
from yarrow.segmented.operations import Operations
from yarrow.decompose.frobenius import frobenius_decomposition

class Functor:
    """ A base class for implementing strict symmetric monoidal hypergraph functors.

    A Functor must implement two things:

    * A mapping on *objects* :py:meth:`yarrow.functor.functor.Functor.map_objects`
    * A mapping on *arrows* :py:meth:`yarrow.functor.functor.Functor.map_arrow`

    The latter is hard to write from scratch!
    Generally you will want to implement
    :py:class:`yarrow.functor.functor.FrobeniusFunctor`.
    Then you only have to implement a mapping on *generating operations*, and the
    `map_arrow` function will be implemented for you.
    """
    # In the paper we denote
    # - map_objects as F₀
    # - map_arrow   as F₁
    @abstractmethod
    def map_objects(self, objects: AbstractFiniteFunction) -> AbstractIndexedCoproduct:
        """ The object map ``F₀`` of a functor ``F : Diagram_Σ → Diagram_Ω`` maps between
        lists of generating objects `F₀ : Σ₀* → Ω₀*`.
        Given an array of generating objects encoded as a FiniteFunction::

            objects : W → Σ₀

        This function must return an *indexed coproduct* representing the lists
        to which each object was mapped::

            sources : W            → Nat
            values  : sum(sources) → Ω₀

        (An "indexed coproduct" is basically a categorical name for a segmented array.)
        """
        ...

    @abstractmethod
    def map_arrow(self, d: AbstractDiagram) -> AbstractDiagram:
        # """The arrow map of a functor """
        """F.map_arrow(d) should apply the functor F to diagram d."""
        ...

def apply_finite_object_map(
        finite_object_map: AbstractIndexedCoproduct,
        wn: AbstractFiniteFunction) -> AbstractIndexedCoproduct:
    """Given an AbstractIndexedCoproduct ``f`` representing a family of ``K`` functions::

        f_i : N_i → Ω₀*, i ∈ K

    where ``f_i`` is the action of a functor ``F`` on generating object ``i``,
    and an object ``wn = L(A)`` in the image of ``L``,
    ``apply_finite_object_map(f, wn)`` computes the object ``F(wn)``
    as a segmented array.
    """
    assert isinstance(finite_object_map, AbstractIndexedCoproduct)
    return type(finite_object_map)(
        values  = finite_object_map.sources.injections(wn) >> finite_object_map.values,
        sources = wn >> finite_object_map.sources)

def map_half_spider(swn: AbstractIndexedCoproduct, f: AbstractFiniteFunction) -> AbstractFiniteFunction:
    """Let ``swn = F.map_objects(f.type[1])`` for some functor ``F``,
    and suppose ``S(f)`` is a half-spider.
    Then ``S(map_half_spider(swn, f)) == F(S(f))``.
    """
    # NOTE: swn should be the result of applying an object map to wn.
    # swn = object_map(wn)
    return swn.sources.injections(f)

def decomposition_to_operations(d: 'AbstractDiagram') -> Operations:
    """ Get the array of operations (and their types) from a Frobenius
    decomposition.  """
    # NOTE: it's *very* important that d is a frobenius decomposition, since we
    # directly use the maps d.G.wi and d.G.wo in the result.
    Fun = d._Fun
    Array = Fun._Array

    # A concrete Fun implementation knows what its IndexedCoproduct class is;
    # see concrete modules yarrow.numpy and yarrow.cupy for details!
    IndexedCoproduct = Fun.IndexedCoproduct

    s_type = IndexedCoproduct(
        sources = Fun(None, bincount(d.G.xi).table),
        values  = d.G.wi >> d.G.wn)

    t_type = IndexedCoproduct(
        sources = Fun(None, bincount(d.G.xo).table),
        values  = d.G.wo >> d.G.wn)

    return Operations(d.G.xn, s_type, t_type)

class FrobeniusFunctor(Functor):
    """ A functor defined in terms of Frobenius decompositions.
    Instead of specifying `map_arrow`, the implementor can specify `map_operations`.
    This should map a *tensoring* of generators to a :py:class:`AbstractDiagram`.
    """
    @abstractmethod
    def map_objects(self, objects: AbstractFiniteFunction) -> AbstractIndexedCoproduct:
        ...

    @abstractmethod
    def map_operations(self, ops: Operations) -> AbstractDiagram:
        """Given an array of generating operations::

            xn : X → Σ₁

        and their types::

            s_type : sum_{i ∈ X} arity(xn(i))   → Σ₀
            t_type : sum_{i ∈ X} coarity(xn(i)) → Σ₀

        You must return an :py:class:`yarrow.diagram.AbstractDiagram` representing the tensoring::

            F₁(xn(0)) ● F₁(xn(1)) ... F₁(xn(X - 1))
        """
        ...

    def map_arrow(self, d: AbstractDiagram) -> AbstractDiagram:
        """ Apply a functor to a diagram. NOTE: You do not need to implement this method """
        Diagram = type(d)
        d = frobenius_decomposition(d)
        ops = decomposition_to_operations(d)

        # swn = F(G(wn))
        # ... is the IndexedCoproduct resulting from applying the functor to the
        # wire labeling d.G.wn
        swn = self.map_objects(d.G.wn)
        h = self.map_operations(ops)

        Fun   = h._Fun
        Graph = h._Graph

        xn = d._Fun.initial(h.G.xn.target, dtype=h.G.xn.table.dtype)

        # build the morphisms (s ; x) ; (id ● h) ; (y ; t) from Proposition B.1
        i = Diagram.identity(swn.values, xn)
        # note: we use the source/target maps of i in constructing those of sx, yt
        # to avoid constructing another array with the same data.
        sx = Diagram(map_half_spider(swn, d.s), i.t + map_half_spider(swn, d.G.wi), Graph.discrete(swn.values, xn))
        yt = Diagram(i.s + map_half_spider(swn, d.G.wo), map_half_spider(swn, d.t), Graph.discrete(swn.values, xn))
        return (sx >> (i @ h) >> yt)


################################################################################
# Built-in functors, supplied as examples.

def identity_object_map(objects: AbstractFiniteFunction) -> AbstractIndexedCoproduct:
    """ The object map of the identity functor """
    Fun = type(objects)
    Array = objects._Array

    IndexedCoproduct = Fun.IndexedCoproduct

    # TODO: write a test for this!
    targets_codomain = None if objects.target is None else objects.target + 1
    return IndexedCoproduct(
        sources = Fun(None, Array.ones(len(objects))),
        values  = objects)

class Identity(Functor):
    """ The identity functor, whose ``map_arrow`` method is implemented by
    actually just returning the same diagram """
    def map_objects(self, objects) -> AbstractIndexedCoproduct:
        """ """
        return identity_object_map(objects)

    def map_arrow(self, d: AbstractDiagram) -> AbstractDiagram:
        """ """
        return d

class FrobeniusIdentity(FrobeniusFunctor):
    """ The identity functor, implemented using Frobenius decompositions.
    This is provided as a simple example of how to use the FrobeniusFunctor type:
    instead of implementing ``map_arrow`` directly, one can instead write a mapping
    on tensorings of operations.
    This is typically much easier, since for a strict monoidal functor F we have
    ``F(f₀ ● f₁ ● ... ● fn) = F(f₀) ● F(f₁) ● ... ● F(fn)``
    """
    def map_objects(self, objects: AbstractFiniteFunction):
        """ """
        return identity_object_map(objects)

    def map_operations(self, ops: Operations) -> AbstractDiagram:
        """ """
        # look up concrete Diagram type from the FiniteFunction
        Diagram = type(ops.xn).Diagram
        return Diagram.tensor_operations(ops)

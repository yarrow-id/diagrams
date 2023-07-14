"""Diagrams are the main datastructure of yarrow, and represent string diagrams.

The :py:class:`AbstractDiagram` is the main datastructure of yarrow.
It represents a string diagram as a *cospan of bipartite multigraphs*.
For example, the diagram below left is represented internally below right:

.. image:: /string-diagram-side-by-side.svg
   :scale: 150%
   :align: center
   :alt: a string diagram and its representation as a cospan

This representation (the :py:class:`AbstractDiagram` class) consists of three
things:

1. An :py:class:`AbstractBipartiteMultigraph` ``G`` (center grey box)
2. The *source map* ``s``: the dotted arrows from the *left* blue box to the center box
3. The *target map* ``t``: the dotted arrows from the *right* blue box to the center box

The center box `G` encodes the internal wiring of the diagram, while ``s`` and
``t`` encode the "dangling wires" on the left and right.

The :py:class:`AbstractDiagram` class is a backend-agnostic implementation.
Concrete implementations choose a *backend*, which is an implementation of the classes
:py:class:`AbstractFiniteFunction` and :py:class:`AbstractBipartiteMultigraph`.

For example, numpy-backed diagrams are implemented by the :py:class:`Diagram` class,
which inherits :py:class:`AbstractDiagram` and sets two class members:

- `_Fun = yarrow.array.numpy`
- `_Graph = yarrow.bipartite_multigraph.BipartiteMultigraph`

For more information on backends, see :ref:`backends`.

Summary
-------

.. autosummary::
    :template: class.rst

    AbstractDiagram
"""
from dataclasses import astuple

from yarrow.finite_function import AbstractFiniteFunction
from yarrow.bipartite_multigraph import AbstractBipartiteMultigraph

# for tensor_operations
from yarrow.segmented.operations import Operations

class AbstractDiagram:
    """ Implements diagrams parametrised by an underlying choice of backend.
    To use this class, inherit from it and set class members:

    - ``_Fun`` (finite functions)
    - ``_Graph`` (bipartite multigraphs)

    See for example the :py:class:`Diagram` class, which uses numpy-backed arrays.
    """
    def __init__(self,
                 s: AbstractFiniteFunction,
                 t: AbstractFiniteFunction,
                 G: AbstractBipartiteMultigraph):
        """Construct a :py:class:`AbstractDiagram` from a triple ``(s, t, G)``.

        Description

        Args:
            s: Finite function of type `A → G.W`
            t: Finite function of type `B → G.W`
            G: An :py:class:`AbstractBipartiteMultigraph`
        """
        self.s = s
        self.t = t
        self.G = G

        # the cospan (s, t) is a pair of arrows
        #     s   t
        #   A → G(W) ← B
        # so we need to verify types work out.
        assert G.W == s.target
        assert G.W == t.target

        # Lastly, the underlying finite function type should be the same.
        _Fun = type(self)._Fun
        assert _Fun == type(s)
        assert _Fun == type(t)
        assert _Fun == G._Fun

    @property
    def wires(self):
        """
        Return the number of 'wires' in the diagram.
        A wire is a node in the graph corresponding to a wire of the string diagram.
        """
        return self.G.W

    @property
    def operations(self):
        """Return the number of generating operations in the diagram."""
        return self.G.X

    @property
    def shape(self):
        """ Return the arity and coarity of the diagram. """
        return self.s.source, self.t.source

    @property
    def type(self):
        """ Return a pair of finite functions representing the type of the morphism.

        Returns:
            (tuple): tuple of:
                source(AbstractFiniteFunction): typed `self.s.domain → Σ₀`
                target(AbstractFiniteFunction): typed `self.t.domain → Σ₀`
        """
        wire_labels = self.G.wn
        return (self.s >> wire_labels, self.t >> wire_labels)

    def __eq__(f, g):
        return f.s == g.s and f.t == g.t and f.G == g.G

    @classmethod
    def empty(cls, wn : AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """
        Args:
            wn: A FiniteFunction typed `0 → Σ₀`: giving the generating objects
            xn: A FiniteFunction typed `0 → Σ₁`: giving the generating operations

        Returns:
            The empty diagram for the monoidal signature (Σ₀, Σ₁)

        Note that for a non-finite signature, we allow the targets of ``wn`` and
        ``xn`` to be ``None``.
        """
        s = t = cls._Fun.initial(0)
        return cls(s, t, cls._Graph.empty(wn, xn))

    @classmethod
    def identity(cls, wn: AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """
        Args:
            wn: A FiniteFunction typed `W → Σ₀`: giving the generating objects
            xn: A FiniteFunction typed `0 → Σ₁`: giving the generating operations

        Returns:
            AbstractDiagram: The identity diagram with `W` wires labeled `wn : W → Σ₀` whose empty set of generators is labeled in Σ₁
        """
        assert xn.source == 0
        s = cls._Fun.identity(wn.source)
        t = cls._Fun.identity(wn.source)
        G = cls._Graph.discrete(wn, xn)
        return cls(s, t, G)

    @classmethod
    def twist(cls, wn_A: AbstractFiniteFunction, wn_B: AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """
        Args:
            wn_A : typed `A → Σ₀`
            wn_B : typed `B → Σ₀`
            xn   : typed `0 → Σ₁`

        Returns:
            AbstractDiagram: The symmetry diagram `σ : A ● B → B ● A`.
        """
        assert xn.source == 0
        wn = wn_A + wn_B
        s = cls._Fun.identity(wn.source)
        t = cls._Fun.twist(wn_A.source, wn_B.source)
        G = cls._Graph.discrete(wn, xn)
        return cls(s, t, G)

    @classmethod
    def spider(cls,
               s: AbstractFiniteFunction,
               t: AbstractFiniteFunction,
               w: AbstractFiniteFunction,
               x: AbstractFiniteFunction):
        """Create a *Frobenius Spider* (see Definition 2.8, Proposition 4.7 of :cite:p:`dpafsd`).

        Args:
            s : source map typed `S → W`
            t : target map typed `T → W`
            w : wires typed `W → Σ₀`
            x : empty set of operations `0 → Σ₁`

        Returns:
            AbstractDiagram: A frobenius spider with `S` inputs and `T` outputs.
        """
        assert x.source == 0
        assert w.source == s.target
        assert w.source == t.target
        G = cls._Graph.discrete(w, x)
        return cls(s, t, G)

    @classmethod
    def half_spider(cls,
                    s: AbstractFiniteFunction,
                    w: AbstractFiniteFunction,
                    x: AbstractFiniteFunction):
        """ Create a *Frobenius Half-Spider*, which is a spider whose target map is the identity """
        # s : A → W
        # w : W → Σ₀
        assert s.target == w.source
        t = cls._Fun.identity(w.source)
        return cls.spider(s, t, w, x)

    def dagger(self):
        """Swap the *source* and *target* maps of the diagram.

        Returns:
            AbstractDiagram: The dagger functor applied to this diagram.
        """
        return type(self)(self.t, self.s, self.G)

    @classmethod
    def singleton(cls, a: AbstractFiniteFunction, b: AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """ Construct a diagram consisting of a single operation (Definition 4.9, :cite:p:`dpafsd`).

        Args:
            x: A single operation represented as an AbstractFiniteFunction of type `1 → Σ₁`
            a: The input type of `x` as a finite function `A → Σ₀`
            b: The output type of `x` as a finite function `B → Σ₀`

        Returns:
            AbstractDiagram: a diagram with a single generating operation.
        """
        Array = cls._Fun._Array
        F = cls._Fun
        assert F == type(a)
        assert F == type(b)
        assert F == type(xn)

        # x : 1 → Σ₁
        assert xn.source == 1

        # Must be able to take coproduct a + b because
        #   wn : A + B → Σ₀
        assert a.target == b.target

        # wi : A → A + B     wo : B → A + B
        wi = F.inj0(a.source, b.source)
        wo = F.inj1(a.source, b.source)

        G = cls._Graph(
            wi=wi,
            wo=wo,

            # xi : A → 1         xo : B → 1
            xi = F.terminal(a.source),
            xo = F.terminal(b.source),

            # wn : A + B → Σ₀
            wn = a + b,

            # pi' : A → Nat       po' : B → Nat
            # pi = pi';ι          po  = po';ι
            pi = F(None, Array.arange(a.source)),
            po = F(None, Array.arange(b.source)),

            xn = xn,
        )

        # Note: s=inj0, t=inj1, so we just reuse wi and wo.
        return cls(s=wi, t=wo, G=G)

    def tensor(f: 'AbstractDiagram', g: 'AbstractDiagram'):
        """Stack one diagram atop another, so `f.tensor(g)` is the diagram depicted by

        .. image:: /tensor-f-g.svg
           :scale: 150%
           :align: center
           :alt: a depiction of the tensor product of diagrams

        Args:
            g(AbstractDiagram): An arbitrary diagram

        Returns:
            AbstractDiagram: The tensor product of this diagram with `g`.
        """
        return type(f)(
            s = f.s @ g.s,
            t = f.t @ g.t,
            G = f.G @ g.G)

    def __matmul__(f, g):
        """ Shorthand for :py:meth:`yarrow.Diagram.tensor`.
        f @ g == f.tensor(g)
        """
        return f.tensor(g)

    def compose(f: 'AbstractDiagram', g: 'AbstractDiagram'):
        """Compose this diagram with `g`, so `f.compose(g)` is the diagram

        .. image:: /compose-f-g.svg
           :scale: 150%
           :align: center
           :alt: a depiction of the tensor product of diagrams


        Args:
            g(AbstractDiagram): A diagram with `g.type[0] == self.type[1]`

        Returns:
            AbstractDiagram: The tensor product of this diagram with `g`.

        Raises:
            AssertionError: If `g.type[0] != f.type[1]`
        """

        assert f.type[1] == g.type[0]
        h = f @ g
        q = f.t.inject0(g.G.W).coequalizer(g.s.inject1(f.G.W))
        return type(f)(
            s = f.s.inject0(g.G.W) >> q,
            t = g.t.inject1(f.G.W) >> q,
            G = h.G.coequalize_wires(q))

    def __rshift__(f, g):
        return f.compose(g)

    @classmethod
    def tensor_list(cls, ds: 'List[AbstractDiagram]', wn=None, xn=None):
        """ Compute the tensor product of a list of diagrams. O(n) time in the size of the result.

        .. warning::
            Does not speed up to O(log n) in the parallel case
        """
        if len(ds) == 0:
            assert wn is not None
            assert xn is not None
            return cls.empty(wn, xn)

        assert wn is None
        assert xn is None
        s = cls._Fun.tensor_list([d.s for d in ds])
        t = cls._Fun.tensor_list([d.t for d in ds])
        G = cls._Graph.coproduct_list([d.G for d in ds])
        return cls(s, t, G)

    @classmethod
    def tensor_operations(cls, ops: Operations):
        pass # hide the docstring for now
        """ Compute the X-fold tensoring of operations

        .. code-block:: text

            xn : X → Σ₁

        whose typings are given by the segmented finite functions

        .. code-block:: text

            s_type : sum_{i ∈ X} arity(xn(i))   → Σ₀
            t_type : sum_{i ∈ X} coarity(xn(i)) → Σ₀

        See Proposition 4.13 in :cite:t:`dpafsd`.
        """
        Fun   = cls._Fun
        Array = Fun._Array

        xn, s_type, t_type = ops.xn, ops.s_type, ops.t_type

        r = Array.arange(0, xn.source)
        # TODO: FIXME: redundant computation.
        # we do a sum of s_type/t_type sources, but we already do a cumsum in
        # segmented_arange, so this is wasted effort!
        Ki = Array.sum(s_type.sources.table)
        Ko = Array.sum(t_type.sources.table)

        i0 = Fun.inj0(Ki, Ko)
        i1 = Fun.inj1(Ki, Ko)

        return cls(
            s = i0,
            t = i1,
            G = cls._Graph(
                xn = xn,
                # Tensor product of terminal maps
                # e.g. 1 1 1 | 2 2 | 3 3 3 3 ...
                xi = Fun(xn.source, Array.repeat(r, s_type.sources.table)),
                xo = Fun(xn.source, Array.repeat(r, t_type.sources.table)),
                # Coproduct of ι₀ maps
                # e.g. 0 1 2 | 0 1 | 0 1 2 3 ...
                pi = Fun(None, Array.segmented_arange(s_type.sources.table)),
                po = Fun(None, Array.segmented_arange(t_type.sources.table)),
                # wires: sources first, then targets
                wi = i0,
                wo = i1,
                wn = s_type.values + t_type.values))

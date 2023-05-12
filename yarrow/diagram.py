from dataclasses import astuple

from yarrow.finite_function import AbstractFiniteFunction, FiniteFunction
from yarrow.bipartite_multigraph import BipartiteMultigraph, AbstractBipartiteMultigraph

# for tensor_operations
from yarrow.segmented.finite_function import AbstractSegmentedFiniteFunction
from yarrow.segmented.operations import Operations

class AbstractDiagram:
    """ Defines a class of Diagram implementations parametrised over underlying
    implementations of:
        * _Fun (finite function)
        * _Graph (bipartite multigraph)
    """
    def __init__(self, s, t, G):
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
        """ Return the arity and coarity of the diagram """
        return self.s.source, self.t.source

    @property
    def type(self):
        """ Return a pair of finite functions representing the type of the morphism.
            source : self.s.domain → Σ₀
            target : self.t.domain → Σ₀
        """
        wire_labels = self.G.wn
        return (self.s >> wire_labels, self.t >> wire_labels)

    def __eq__(f, g):
        return f.s == g.s and f.t == g.t and f.G == g.G

    @classmethod
    def empty(cls, wn : AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """ Return the empty diagram for a signature.

        :param `wn : 0 → Σ₀`: A FiniteFunction giving the generating objects
        :param `xn : 0 → Σ₁`: A FiniteFunction giving the generating operations

        :return: The empty diagram for the monoidal signature (Σ₀, Σ₁)
        """
        s = t = cls._Fun.initial(0)
        return cls(s, t, cls._Graph.empty(wn, xn))

    @classmethod
    def identity(cls, wn: AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """
        Create the identity diagram with n wires labeled
            wn : W → Σ₀
        whose (empty set of) generators are labeled in Σ₁
            xn : 0 → Σ₁
        """
        assert xn.source == 0
        s = cls._Fun.identity(wn.source)
        t = cls._Fun.identity(wn.source)
        G = cls._Graph.discrete(wn, xn)
        return cls(s, t, G)

    @classmethod
    def twist(cls, wn_A: AbstractFiniteFunction, wn_B: AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """
        Given functions
            wn_A : A → Σ₀
            wn_B : B → Σ₀
            xn   : 0 → Σ₁
        Return the symmetry diagram of type A ● B → B ● A.
        """
        assert xn.source == 0
        wn = wn_A + wn_B
        s = cls._Fun.identity(wn.source)
        t = cls._Fun.twist(wn_A.source, wn_B.source)
        G = cls._Graph.discrete(wn, xn)
        return Diagram(s, t, G)

    @classmethod
    def spider(cls,
               s: AbstractFiniteFunction,
               t: AbstractFiniteFunction,
               w: AbstractFiniteFunction,
               x: AbstractFiniteFunction):
        """ Given
            s : S → W
            t : T → W
            w : W → Σ₀
            x : 0 → Σ₁
        Construct the Frobenius spider (s, t, Discrete(w))
        """
        assert x.source == 0
        assert w.source == s.target
        assert w.source == t.target
        G = cls._Graph.discrete(w, x)
        return Diagram(s, t, G)

    def dagger(self):
        return Diagram(self.t, self.s, self.G)

    @classmethod
    def singleton(cls, a: AbstractFiniteFunction, b: AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """ Given a generator x and a typing (A, B)
            x : 1 → Σ₁
            a : A → Σ₀
            b : B → Σ₀
        Construct the singleton diagram of x.
        """
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

            # pi : A → Nat       po : B → Nat
            pi = F.identity(a.source),
            po = F.identity(b.source),

            xn = xn,
        )

        # Note: s=inj0, t=inj1, so we just reuse wi and wo.
        return cls(s=wi, t=wo, G=G)

    def tensor(f, g):
        return Diagram(
            s = f.s @ g.s,
            t = f.t @ g.t,
            G = f.G @ g.G)

    def __matmul__(f, g):
        return f.tensor(g)

    def compose(f, g):
        assert f.type[1] == g.type[0]
        h = f @ g
        q = f.t.inject0(g.G.W).coequalizer(g.s.inject1(f.G.W))
        return Diagram(
            s = f.s.inject0(g.G.W) >> q,
            t = g.t.inject1(f.G.W) >> q,
            G = h.G.coequalize_wires(q))

    def __rshift__(f, g):
        return f.compose(g)

    @classmethod
    def tensor_operations(cls, ops: Operations):
        """ Compute the X-fold tensoring of operations

            xn : X → Σ₁

        whose typings are given by the segmented finite functions

            s_type : sum_{i ∈ X} arity(xn(i))   → Σ₀
            t_type : sum_{i ∈ X} coarity(xn(i)) → Σ₀

        (This is Proposition 4.13 in the paper)
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
            G = Diagram._Graph(
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

class Diagram(AbstractDiagram):
    """ The default Yarrow diagram type uses numpy-backed finite functions and bipartite multigraphs """
    _Fun = FiniteFunction
    _Graph = BipartiteMultigraph

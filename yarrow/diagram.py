from yarrow.finite_function import AbstractFiniteFunction, FiniteFunction
from yarrow.bipartite_multigraph import BipartiteMultigraph, AbstractBipartiteMultigraph

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
        return self.G.wires

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

    @classmethod
    def empty(cls):
        """ The empty diagram """
        s = t = cls._Fun.initial(0)
        return cls(s, t, cls._Graph.empty())

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
    def singleton(cls, a: AbstractFiniteFunction, b: AbstractFiniteFunction, x: AbstractFiniteFunction):
        """ Given a generator x and a typing (A, B)
            x : 1 → Σ₁
            a : A → Σ₀
            b : B → Σ₀
        Construct the singleton diagram of x.
        """
        F = cls._Fun
        assert F == type(a)
        assert F == type(b)
        assert F == type(x)

        # wn : A + B → Σ₀
        assert a.target == b.target
        wn = a + b

        # x : 1 → Σ₁
        xn = x

        # wi : A → A + B     wo : B → A + B
        wi = F.inj0(a.source, b.source)
        wo = F.inj1(a.source, b.source)

        # xi : A → 1         xo : B → 1
        xi = F.terminal(a.source)
        xo = F.terminal(b.source)

        # pi : A → Nat       po : B → Nat
        pi = F.identity(a.source)
        po = F.identity(b.source)

        G = cls._Graph(wi, wo, xi, xo, wn, pi, po, xn)
        return cls(wi, wo, G)

class Diagram(AbstractDiagram):
    """ The default Yarrow diagram type uses numpy-backed finite functions and bipartite multigraphs """
    _Fun = FiniteFunction
    _Graph = BipartiteMultigraph

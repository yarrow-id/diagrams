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
        s = cls._Fun.identity(wn.source)
        t = cls._Fun.identity(wn.source)
        G = cls._Graph.discrete(wn, xn)
        return cls(s, t, G)

class Diagram(AbstractDiagram):
    """ The default Yarrow diagram type uses numpy-backed finite functions and bipartite multigraphs """
    _Fun = FiniteFunction
    _Graph = BipartiteMultigraph

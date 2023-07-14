"""A representation of the "internal wiring" of string diagrams.
Bipartite multigraphs have edge labels corresponding to the "ports" of
operations, and node labels either generating objects or generating operations
of a signature Σ.

As with other classes, these graphs are implemented with an abstract base class
:py:class:`AbstractBipartiteMultigraph`,
whose concrete instantiations choose a backend.
For example, :py:class:`BipartiteMultigraph` are backed by numpy arrays.
"""
from dataclasses import dataclass
from yarrow.finite_function import AbstractFiniteFunction

class AbstractBipartiteMultigraph:
    """ The type of bipartite multigraphs, parametrised by cls._Fun, the
    underlying representation of finite functions """
    def __init__(self, wi, wo, xi, xo, wn, pi, po, xn):
        """Create a BipartiteMultigraph from its component finite functions.
        For more details see :cite:p:`dpafsd`, Section 3.2.
        """
        # Edge/Wire incidence
        self.wi = wi
        self.wo = wo

        # Edge/Generator incidence
        self.xi = xi
        self.xo = xo

        # Wire + generator labels
        self.xn = xn
        self.wn = wn

        # Port labels
        self.pi = pi
        self.po = po

        # Check schema of bipartite multigraphs is satisfied
        if wi.target != wo.target:
            raise ValueError("wi.target != wo.target")
        if xi.target != xo.target:
            raise ValueError("xi.target != xo.target")
        if wi.source != xi.source:
            raise ValueError("wi.source != xi.source")
        if wo.source != xo.source:
            raise ValueError("wo.source != xo.source")
        if wn.source != wi.target:
            raise ValueError("wn.source != wi.target")
        if wn.source != wi.target:
            raise ValueError("wn.source != wi.target")
        if pi.source != xi.source:
            raise ValueError("pi.source != xi.source")
        if po.source != xo.source:
            raise ValueError("pi.source != xo.source")
        if xn.source != xi.target:
            raise ValueError("xn.source != xi.target")

    @property
    def W(self):
        """Test

        Returns:
            G(W)
        """
        # wn : G(W) → Σ₀
        return self.wn.source

    @property
    def Ei(self):
        """Test

        Returns:
            The number of *input edges* in the graph
        """
        return self.wi.source

    @property
    def Eo(self):
        """
        Returns:
            The number of *output edges* in the graph
        """
        return self.wo.source

    @property
    def X(self):
        """
        Returns:
            int: Corresponds to G(X), the number of generating operations in the diagram"""
        # xn : G(X) → Σ₁
        return self.xn.source

    @classmethod
    def empty(cls, wn, xn):
        """
        Args:
            wn: Finite function typed `0 → Σ₀`
            xn: Finite function typed `0 → Σ₁`

        Returns:
            AbstractBipartiteMultigraph: The empty bipartite multigraph with no edges and no nodes.
        """
        assert wn.source == 0
        assert xn.source == 0
        e = cls._Fun.initial(0)
        pi = po = cls._Fun.initial(None)
        return cls(e, e, e, e, wn, pi, po, xn)

    @classmethod
    def discrete(cls, wn: AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """
        Create the discrete graph of n wires for a given monoidal signature Σ
        whose maps are all initial except for `wn`.

        Args:
            wn: An array of wire labels as a finite function typed `n → Σ₀`
            xn: The type of operations as an empty finite function typed `0 → Σ₁`
        """
        if xn.source != 0:
            raise ValueError("xn.source != 0")

        return cls(
            # There are no edges, so we make empty maps for all edge data
            wi = cls._Fun.initial(wn.source),
            wo = cls._Fun.initial(wn.source),
            xi = cls._Fun.initial(0),
            xo = cls._Fun.initial(0),

            # TODO: dirty hack alert: None represents any "non-finite" codomain here.
            # In this case, we need edge_data : E → Nat
            # but this (obviously) cannot be represented by a finite function.
            # This is justified because both these maps factor through some
            # finite function: we just don't know what it is at this point in
            # the code.
            pi = cls._Fun.initial(None),
            po = cls._Fun.initial(None),

            wn = wn, # there are w_label.target wires
            xn = xn, # and no operation nodes
        )

    def __eq__(a, b):
        return \
            a.wi == b.wi and \
            a.wo == b.wo and \
            a.xi == b.xi and \
            a.xo == b.xo and \
            a.wn == b.wn and \
            a.pi == b.pi and \
            a.po == b.po and \
            a.xn == b.xn

    def coproduct(f, g):
        """Compute the coproduct of bipartite multigraphs

        Args:
            g: an arbitrary AbstractBipartiteMultigraph over the same signature

        Returns:
            The coproduct ``self + g``.
        """
        # check signatures match
        assert f.wn.target == g.wn.target
        assert f.xn.target == g.xn.target

        return type(f)(
            # Tensor product of data
            wi=f.wi @ g.wi,
            wo=f.wo @ g.wo,
            xi=f.xi @ g.xi,
            xo=f.xo @ g.xo,
            # Coproduct of attributes
            wn=f.wn + g.wn,
            pi=f.pi + g.pi,
            po=f.po + g.po,
            xn=f.xn + g.xn,
        )

    def __matmul__(f, g):
        return f.coproduct(g)

    @classmethod
    def coproduct_list(cls, Gs: 'List[AbstractBipartiteMultigraph]', wn=None, xn=None):
        """ Compute the coproduct of a list of bipartite multigraphs. O(n) in the size of the result.

        .. warning::
            Does not speed up to O(log n) in the parallel case.
        """
        if len(Gs) == 0:
            assert wn is not None
            assert xn is not None
            return cls.empty(wn, xn)

        # can't specify Σ if Gs is non-empty
        assert wn is None
        assert xn is None
        return cls(
            wi=cls._Fun.tensor_list([g.wi for g in Gs]),
            wo=cls._Fun.tensor_list([g.wo for g in Gs]),
            xi=cls._Fun.tensor_list([g.xi for g in Gs]),
            xo=cls._Fun.tensor_list([g.xo for g in Gs]),
            wn=cls._Fun.coproduct_list([g.wn for g in Gs]),
            pi=cls._Fun.coproduct_list([g.pi for g in Gs]),
            po=cls._Fun.coproduct_list([g.po for g in Gs]),
            xn=cls._Fun.coproduct_list([g.xn for g in Gs]))

    # Apply a morphism α of bipartite multigraphs whose only
    # non-identity component is α_W = q.
    def coequalize_wires(self, q : AbstractFiniteFunction):
        """
        Apply a morphism α of bipartite multigraphs
        whose only non-identity component α_W = q
        for some coequalizer q.

        Args:
            q: An AbstractFiniteFunction which is a coequalizer.

        Returns:
            AbstractBipartiteMultigraph: The bipartite multigraph equal to the target of α.
        """
        assert self.wn.source == q.source
        u = universal(q, self.wn)

        # Check that resulting diagram commutes
        # TODO: this is unnecessary extra computation when the user knows that q is a coequalizer.
        # Make a flag?
        Array = type(q)._Array
        if not (q >> u) == self.wn:
            raise ValueError(f"Universal morphism doesn't make {q};{u}, {self.wn} commute. Is q really a coequalizer?")

        return type(self)(
                wi=self.wi >> q,
                wo=self.wo >> q,
                wn=u,
                xi=self.xi,
                xo=self.xo,
                pi=self.pi,
                po=self.po,
                xn=self.xn)

# Let G be a bipartite multigraph.
# Given a coequalizer q : G(W) → Q of finite functions,
# Define the map
#   α : G → G'
# with
#   α_W = q
#   α_Y = id_Y
#
# Then we need a map
#   Q(W) → Σ₀
# which we can get because
#   G(wn) : G(W) → Σ₀
# coequalizes, and so there exists some unique
#   u : Q(W) → Σ₀
# such that
#   q ; u = G(wn)
# And this can be computed as follows:
#   u[q[i]] = G(wn)
def universal(q: AbstractFiniteFunction, f: AbstractFiniteFunction):
    """
    Given a coequalizer q : B → Q of morphisms a, b : A → B
    and some f : B → B' such that f(a) = f(b),
    Compute the universal map u : Q → B'
    such that q ; u = f.
    """
    target = f.target
    u = q._Array.zeros(q.target, dtype=f.table.dtype)
    # TODO: in the below we assume the PRAM CRCW model: multiple writes to the
    # same memory location in the 'u' array can happen in parallel, with an
    # arbitrary write succeeding.
    # Note that this doesn't affect correctness because q and f are co-forks,
    # and q is a coequalizer.
    # However, this won't perform well on e.g., GPU hardware. FIXME!
    u[q.table] = f.table
    return type(f)(target, u)

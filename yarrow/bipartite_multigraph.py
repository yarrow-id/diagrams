from dataclasses import dataclass
from yarrow.finite_function import AbstractFiniteFunction, FiniteFunction

class BipartiteMultigraph:
    """ The type of bipartite multigraphs, parametrised by cls._Fun, the
    underlying representation of finite functions """
    def __init__(self, wi, wo, xi, xo, wn, pi, po, xn):
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
        """ Returns G(W) """
        # wn : G(W) → Σ₀
        return self.wn.source

    @property
    def X(self):
        """ Returns G(X) """
        # xn : G(X) → Σ₁
        return self.xn.source

    @classmethod
    def empty(cls):
        """ Construct the empty bipartite multigraph with no edges and no nodes """
        e = cls._Fun.initial(0)
        return BipartiteMultigraph(e, e, e, e, e, e, e, e)

    @classmethod
    def discrete(cls, wn: AbstractFiniteFunction, xn: AbstractFiniteFunction):
        """
        Create the discrete graph of n wires for a given monoidal signature Σ
        whose maps are all initial except
            wn : n → Σ₀
        and we have
            xn : 0 → Σ₁
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

    # Apply a morphism α of bipartite multigraphs whose only
    # non-identity component is α_W = q.
    def coequalize_wires(self, q : AbstractFiniteFunction):
        """
        Apply a morphism α of bipartite multigraphs
        whose only non-identity component α_W = q
        for some coequalizer q.
        """
        assert self.wn.source == q.source
        u = universal(q, self.wn)
        return BipartiteMultigraph(
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
    u = q._Array.zeros(q.target)
    # TODO: in the below we assume the PRAM CRCW model: multiple writes to the
    # same memory location in the 'u' array can happen in parallel, with an
    # arbitrary write succeeding.
    # Note that this doesn't affect correctness because q and f are co-forks,
    # and q is a coequalizer.
    # However, this won't perform well on e.g., GPU hardware. FIXME!
    u[q.table] = f.table
    return type(f)(target, u)

class NumpyBipartiteMultigraph(BipartiteMultigraph):
    _Fun = FiniteFunction

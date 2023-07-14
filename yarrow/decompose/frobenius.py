from yarrow.diagram import AbstractDiagram

def frobenius_decomposition(d: AbstractDiagram) -> AbstractDiagram:
    """ Given a Diagram, permute its xi, xo, pi, and po maps to be in
    (generator, port) order.
    """
    # A Frobenius Decomposition is really just diagram whose edges are put in
    # "generator, port" order.
    # Obtaining the half spiders and tensorings from such a diagram is trivial:
    # - s, t  are the source, targets of the diagram
    # - everything else is just a component of the bipartite multigraph
    p = sort_x_p(d.G.xi, d.G.pi)
    q = sort_x_p(d.G.xo, d.G.po)

    return type(d)(
        s  = d.s,
        t  = d.t,
        G  = type(d.G)(
            wi = p >> d.G.wi, # e_s
            wo = q >> d.G.wo, # e_t
            wn = d.G.wn,

            xi = p >> d.G.xi,
            xo = q >> d.G.xo,
            pi = p >> d.G.pi,
            po = q >> d.G.po,
            xn = d.G.xn)
    )

def sort_x_p(x, port):
    # Sort by the compound key <x, port>.
    # First argsorts port ("p")
    # then stably argsorts by x
    Array = x._Array
    assert Array == port._Array
    # x, port must be equal length arrays
    assert x.source == port.source

    p = Array.argsort(port.table)
    table = Array.argsort(x.table[p])
    return type(x)(x.source, table[p])

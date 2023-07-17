""" Layered decomposition for numpy-backed diagrams.
Note that this (currently) uses SciPy sparse arrays, so it can't be used for diagrams backed
by other array libraries (e.g., CuPy).

Use the ``layer`` function to assign a *layering* to operations in the diagram.
This is like a topological sort, except multiple operations can be assigned to
the same layering.
"""
import numpy as np
import scipy.sparse as sp

from yarrow.numpy import *

def make_sparse(s: FiniteFunction, t: FiniteFunction):
    """Given finite functions ``s : E → A`` and ``t : E → B``
    representing a bipartite graph ``G : A → B``,
    return the sparse ``B×A`` adjacency matrix representing ``G``.
    """
    assert s.source == t.source
    N = s.source
    # (data, (row, col))
    # rows are *outputs*, so row = t.table
    # cols are *inputs* so col = s.table
    return sp.csr_array((np.ones(N, dtype=bool), (t.table, s.table)), shape=(t.target, s.target))

def operation_adjacency(d: Diagram):
    """ Construct the underlying graph of operation adjacency from a diagram.
    An operation ``x`` is adjacent to an operation ``y`` if there is a directed
    path from ``x`` to ``y`` going through a single ■-node.
    """
    # construct the adjacency matrix for generators
    Mi = make_sparse(d.G.wi, d.G.xi)
    Mo = make_sparse(d.G.wo, d.G.xo)
    return Mi @ Mo.T

# Kahn's Algorithm, but vectorised a bit.
# https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
# Attempts to 'parallelize' the layering where possible, e.g.:
#
#          ○--\
#              ○---○
#          ○--/
#
#       ---○--------
#
#  layer   0   1   2
#
def kahn(adjacency: sp.csr_array):
    """ A version of Kahn's algorithm which assigns a *layering* to each ○-node,
    but where multiple nodes can have the same layering.

    Returns a pair of arrays ``(order, visited)``.
    ``order[v]`` is a natural number indicating the computed ordering of node ``v``,
    and ``visited[v]`` is 1 if and only if ``v`` was visited while traversing the graph.

    If not all vertices were visited, the graph had a cycle.
    """
    n, m = adjacency.shape
    assert n == m
    adjacency = adjacency.astype(int)

    # NOTE: convert to numpy ndarray instead of matrix given by adjacency;
    # this makes indexing behaviour a bit nicer!
    # NOTE: we use reshape here because adjacency.sum() gives different dimensions when input is 0x0!
    indegree = np.asarray(adjacency.sum(axis=1, dtype=int)).reshape((n,))

    # return values
    visited  = np.zeros(n, dtype=bool)
    order    = np.zeros(n, dtype=int)

    # start at nodes with no incoming edges
    start = (indegree == 0).nonzero()[0]

    # initialize the frontier at the requested start nodes
    k = len(start)
    frontier = sp.csr_array((np.ones(k, int), (start, np.zeros(k, int))), (n, 1))

    # as long as the frontier contains some nodes, we'll keep going.
    depth = 0
    while frontier.nnz > 0:
        # Mark nodes in the current frontier as visited,
        # and set their layering value ('order') to the current depth.
        frontier_ixs = frontier.nonzero()[0]
        visited[frontier_ixs] = True
        order[frontier_ixs] = depth

        # Find "reachable", which is the set of nodes adjacent to the current frontier.
        # Decrement the indegree of each adjacent node by the number of edges between it and the frontier.
        # NOTE: nodes only remain in the frontier for ONE iteration, so this
        # will only decrement once for each edge.
        reachable = adjacency @ frontier
        reachable_ix = reachable.nonzero()[0]
        indegree[reachable_ix] -= reachable.data

        # Compute the new frontier: the reachable nodes with indegree equal to zero (no more remaining edges!)
        # NOTE: indegree is an (N,1) matrix, so we select out the first row.
        new_frontier_ixs = reachable_ix[indegree[reachable_ix] == 0]
        k = len(new_frontier_ixs)
        frontier = sp.csr_array((np.ones(k, int), (new_frontier_ixs, np.zeros(k, int))), shape=(n, 1))

        # increment depth so the new frontier will be layered correctly
        depth += 1

    # Return the layering (order) and whether each node was visited.
    # Note that if not np.all(visited), then there must be a cycle.
    return order, visited

def layer(d: Diagram):
    """ Assign a *layering* to a diagram ``d``.
        This computes a FiniteFunction ``f : G(X) → K``,
        mapping ○-nodes of ``d.G`` to a natural number in the ``range(0, K)``.
        This mapping is 'dense' in the sense that for each ``i ∈ {0..K}``,
        there is always some ○-node v for which ``f(v) = i``.
    """
    # (Partially) topologically sort it using a kahn-ish algorithm
    # NOTE: if completed is not all True, then some generators were not ordered.
    # In this case, we leave their ordering as 0, since they contain cycles.
    M = operation_adjacency(d)
    ordering, completed = kahn(M)
    return FiniteFunction(d.operations, ordering), completed

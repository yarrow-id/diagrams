"""
A shim around the numpy module that exposes common array operations.
Note that we don't just expose the numpy module for two reasons:
    (1) We need an additional function (connected_components)
    (2) We want to document the underlying primitive functions required of the
    backing array library.
"""
import numpy as np
import scipy.sparse as sparse

DEFAULT_DTYPE='int64'

def array(*args, **kwargs):
    return np.array(*args, **kwargs)

def max(*args, **kwargs):
    return np.max(*args, **kwargs)

def arange(*args, **kwargs):
    return np.arange(*args, **kwargs)

def all(*args, **kwargs):
    return np.all(*args, **kwargs)

def zeros(*args, **kwargs):
    return np.zeros(*args, **kwargs)

def ones(*args, **kwargs):
    return np.ones(*args, **kwargs)

def concatenate(*args, **kwargs):
    return np.concatenate(*args, **kwargs)

# Compute the connected components of a graph.
# connected components of a graph, encoded as a list of edges between points
# so we have s, t arrays encoding edges (s[i], t[i]) of a square n×n matrix.
# NOTE: we have to wrap libraries since we don't tend to get a consistent interface,
# and don't want to expose e.g. sparse graphs in the main code.
def connected_components(source, target, n, dtype=DEFAULT_DTYPE):
    """
    Compute the connected components of an n×n graph,
    encoded as a list of edges (source[i] → target[i])
    """
    if len(source) != len(target):
        raise ValueError("Expected a graph encoded as a pair of arrays (source, target) of the same length")

    assert len(source) == len(target)

    # make an n×n sparse matrix representing the graph with edges
    # source[i] → target[i]
    ones = np.ones(len(source), dtype=DEFAULT_DTYPE)
    M = sparse.csr_matrix((ones, (source, target)), shape=(n, n))

    # compute & return connected components
    c, cc_ix = sparse.csgraph.connected_components(M)
    return c, cc_ix

import scipy.sparse as sp
import numpy as np

# SUT
from yarrow.finite_function import bincount
from yarrow.numpy import Diagram
from yarrow.numpy.layer import layer, kahn, operation_adjacency

from tests.strategies import diagrams

from hypothesis import given
import hypothesis.strategies as st

# TODO:
# Generators:
#   - monogamous acyclic diagrams 
# Tests:
#   - Check that layer(ma_diagram) visits all operations
#   - Check that for MA diagrams d₁ and d₂ and layer(d₁ ; d₂) ...
#       - d₁ layerings should be unchanged
#       - d₂ layerings should all be larger or equal to layer(d₂)
#           - NOTE: relies on operations not being reordered after composition

# Generate a random acyclic adjacency matrix
_MAX_MATRIX_SIZE = 512
@st.composite
def acyclic_adjacency_matrix(draw):
    N = draw(st.integers(min_value=0, max_value=_MAX_MATRIX_SIZE))

    # generate a random matrix, zero below diagonal
    # Density set to 1/N, so memory usage is O(N).
    density = 1 if N == 0 else 1/N

    # NOTE: use LIL format so we can call setdiag efficiently below
    M = sp.random(N, N, density=density, format='lil', dtype=bool)

    # Set diagonal to zero otherwise we can get self-loops!
    M.setdiag(0)

    # convert to CSR for later.
    M = sp.csr_array(M)

    # Return a matrix which zeroes out the upper triangle
    return sp.tril(M.astype(int))

# Test kahn layering visits all nodes in an acyclic matrix
@given(acyclic_adjacency_matrix())
def test_kahn_acyclic_all_visited(M):
    """ Verify that kahn layering correctly orders nodes and visits all nodes in
    an acyclic graph """
    order, visited = kahn(M)

    # All nodes should be visited (the graph is acyclic)
    assert np.all(visited)

    # TODO: check that the ordering reflects the acyclic structure.
    pass

# Test kahn layering *doesn't* visit all nodes in a *cyclic* matrix.
@given(acyclic_adjacency_matrix())
def test_kahn_cyclic_not_all_visited(M):
    # TODO: this test only checks a small subset of cyclic graphs. Extend the
    # generator to a larger class.

    # put a self-cycle on all nodes
    M.setdiag(1)

    # All nodes should be in layer 0 because there are self-cycles on all nodes
    order, visited = kahn(M)

    # all nodes should have order 0, and none should be visited.
    assert np.all(order == 0)
    assert not np.any(visited)

# Test operation_adjacency function does not crash :)
@given(d=diagrams())
def test_operation_adjacency(d: Diagram):
    M = operation_adjacency(d)

    # NOTE: the (commented) test code below is wrong:
    # Given an operation, we don't know how many others it connects to just by its arity/coarity, because of the Frobenius structure. We might have for each port:
    #   - No other operations connected to ("discard")
    #   - Every other operation connected to ("copy")

    # check number of incoming edges is less than or equal to arity
    # arity = bincount(d.G.xi)
    # indegree = M.sum(axis=1)
    # assert np.all(arity.table >= indegree)

    # check number of outgoing edges is less than or equal to coarity
    # coarity = bincount(d.G.xo)
    # outdegree = M.sum(axis=0)
    # assert np.all(coarity.table >= outdegree)
    pass

# Test the layer function does not crash
@given(d=diagrams())
def test_layer_call(d: Diagram):
    layer(d)

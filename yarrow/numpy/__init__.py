""" NumPy-backed finite functions, bipartite multigraphs, and diagrams.

**Additional NumPy-backend-only code:**

.. autosummary::
    :toctree: _autosummary
    :recursive:

    yarrow.numpy.layer

**NumPy-backed datastructures**:
"""
# Abstract implementations
from yarrow.finite_function import *
from yarrow.bipartite_multigraph import *
from yarrow.diagram import *
from yarrow.segmented.finite_function import AbstractIndexedCoproduct, AbstractSegmentedFiniteFunction

# Array backend
import yarrow.array.numpy as numpy_backend

class FiniteFunction(AbstractFiniteFunction):
    """ NumPy-backed finite functions """
    _Array = numpy_backend

class IndexedCoproduct(AbstractIndexedCoproduct):
    _Fun = FiniteFunction

class BipartiteMultigraph(AbstractBipartiteMultigraph):
    """ NumPy-backed bipartite multigraphs """
    _Fun = FiniteFunction

class Diagram(AbstractDiagram):
    """ NumPy-backed string diagrams """
    _Fun = FiniteFunction
    _Graph = BipartiteMultigraph

class SegmentedFiniteFunction(AbstractSegmentedFiniteFunction):
    _Array = numpy_backend
    _Fun = FiniteFunction

# If we had types, this would be 'type-level function' giving us the
# implementation of each of these classes in terms of the base (Numpy-backed
# FiniteFunction)
FiniteFunction.IndexedCoproduct = IndexedCoproduct
FiniteFunction.BipartiteMultigraph = BipartiteMultigraph
FiniteFunction.Diagram = Diagram

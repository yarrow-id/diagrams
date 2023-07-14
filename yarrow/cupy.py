""" CuPy-backed finite functions, bipartite multigraphs, and diagrams.

.. danger::
   **Experimental Module**

   This code is not thoroughly tested.
   It's included here as a proof-of-concept for GPU acceleration.
   The way backends are selected is also likely to change in the future.
"""
# Abstract implementations
from yarrow.finite_function import *
from yarrow.bipartite_multigraph import *
from yarrow.diagram import *
from yarrow.segmented.finite_function import AbstractIndexedCoproduct, AbstractSegmentedFiniteFunction

# Array backend
import yarrow.array.cupy as cupy

class FiniteFunction(AbstractFiniteFunction):
    """ CuPy-backed finite functions """
    _Array = cupy

class IndexedCoproduct(AbstractIndexedCoproduct):
    _Fun = FiniteFunction

class BipartiteMultigraph(AbstractBipartiteMultigraph):
    """ CuPy-backed bipartite multigraphs """
    _Fun = FiniteFunction

class Diagram(AbstractDiagram):
    """ CuPy-backed string diagrams """
    _Fun = FiniteFunction
    _Graph = BipartiteMultigraph

class SegmentedFiniteFunction(AbstractSegmentedFiniteFunction):
    _Array = cupy
    _Fun = FiniteFunction

# If we had types, this would be 'type-level function' giving us the
# implementation of each of these classes in terms of the base (Numpy-backed
# FiniteFunction)
FiniteFunction.IndexedCoproduct = IndexedCoproduct
FiniteFunction.BipartiteMultigraph = BipartiteMultigraph
FiniteFunction.Diagram = Diagram

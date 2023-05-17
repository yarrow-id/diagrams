""" CuPy-backed finite functions, bipartite multigraphs, and diagrams.

.. danger::
   **Experimental Module**

   This code is not thoroughly tested.
   It's included here as a proof-of-concept for GPU acceleration.
   The way backends are selected is also likely to change in the future.
"""
from yarrow.finite_function import *
from yarrow.bipartite_multigraph import *
from yarrow.diagram import *

import yarrow.array.cupy as cupy

class CupyFiniteFunction(AbstractFiniteFunction):
    """ CuPy-backed finite functions """
    _Array = cupy

class CupyBipartiteMultigraph(AbstractBipartiteMultigraph):
    """ CuPy-backed bipartite multigraphs """
    _Fun = CupyFiniteFunction

class CupyDiagram(AbstractDiagram):
    """ CuPy-backed string diagrams """
    _Fun = CupyFiniteFunction
    _Graph = CupyBipartiteMultigraph

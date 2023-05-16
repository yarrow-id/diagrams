""" CuPy-backed diagrams """
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

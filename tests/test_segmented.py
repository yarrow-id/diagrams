from yarrow.finite_function import FiniteFunction
from yarrow.segmented.finite_function import SegmentedFiniteFunction

from hypothesis import given
import hypothesis.strategies as st

from tests.strategies import *

from hypothesis import settings, reproduce_failure

@given(fsx=common_targets())
def test_indexed_coproduct(fsx):
    """ Test the sequential indexed coproduct against parallel/vectorised one """
    fs, x = fsx

    target = 0 if len(fs) == 0 else fs[0].target
    expected = \
        FiniteFunction.coproduct_list([ fs[x(i)] for i in range(0, x.source) ], target)
    actual = SegmentedFiniteFunction.from_list(fs).coproduct(x)

    assert actual == expected
    assert actual.source == sum(fs[x(i)].source for i in range(0, x.source))
    assert actual.target == target

@given(fsx=finite_function_lists())
def test_indexed_tensor(fsx):
    """ Test the sequential indexed *tensor* against parallel/vectorised one """
    fs, x = fsx
    expected = FiniteFunction.tensor_list([fs[x(i)] for i in range(0, x.source) ])
    actual   = SegmentedFiniteFunction.from_list(fs).tensor(x)
    assert actual == expected
    assert actual.source == sum(fs[x(i)].source for i in range(0, x.source))
    assert actual.target == sum(fs[x(i)].target for i in range(0, x.source))


################################################################################
# Segmented operations

@given(ops=operations())
def test_tensor_operations_type(ops):
    d = Diagram.tensor_operations(ops)

    A, B = d.type
    assert A == ops.s_type.values
    assert B == ops.t_type.values

    # check number of edges and wires in the diagram.
    # Ei should be equal to total input arity
    # Eo equal to total output arity
    # wires = Ei + Eo.
    Ki = len(ops.s_type.values)
    Ko = len(ops.t_type.values)

    assert d.G.W  == Ki + Ko
    assert d.G.Ei == Ki
    assert d.G.Eo == Ko
    assert d.G.X  == ops.xn.source

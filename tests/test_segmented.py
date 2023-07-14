from hypothesis import given
import hypothesis.strategies as st

from yarrow.numpy import FiniteFunction, SegmentedFiniteFunction

from tests.strategies import *

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

@given(segmented_finite_functions())
def test_segmented_finite_functions_roundtrip(sff):
    # Given a list of functions, we can convert to a segmented finite function and back losslessly.
    # NOTE: this is not true the other way, because there are multiple valid target values for the 'sources' and 'targets' fields.
    fs = list(sff)
    roundtrip = list(type(sff).from_list(fs))
    assert fs == roundtrip

@given(segmented_finite_functions())
def test_segmented_finite_functions_roundtrip_op(sff):
    # Roundtrip from SFF → list → SFF
    # NOTE: this is not lossless, since sff.sources.target and
    # sff.targets.target values might be lost.
    roundtrip = type(sff).from_list(list(sff))

    # sources must match exactly, but we ignore the domains of finite functions
    # because there are multiple valid choices.
    assert np.all(sff.sources.table == roundtrip.sources.table)
    assert np.all(sff.targets.table == roundtrip.targets.table)
    assert np.all(sff.values.table  == roundtrip.values.table)

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

@given(ops=operations())
def test_operations_iter(ops):
    ops_list = list(ops)
    assert len(ops_list) == len(ops)
    assert np.all(np.array([ x[0] for x in ops_list ], dtype=ops.xn.table.dtype) == ops.xn.table)
    # check types have correct codomain (i.e. Σ₀)
    assert np.all(np.array([ x[1].target for x in ops_list ], dtype=int) == ops.s_type.values.target)
    assert np.all(np.array([ x[2].target for x in ops_list ], dtype=int) == ops.t_type.values.target)

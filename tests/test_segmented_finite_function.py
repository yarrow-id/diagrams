from yarrow.finite_function import FiniteFunction
from yarrow.segmented_finite_function import SegmentedFiniteFunction

from hypothesis import given
import hypothesis.strategies as st

from tests.strategies import *

from hypothesis import settings, reproduce_failure


@given(fsx=common_targets())
def test_indexed_coproduct(fsx):
    fs, x = fsx

    target = 0 if len(fs) == 0 else fs[0].target
    expected = \
        FiniteFunction.coproduct_list([ fs[x(i)] for i in range(0, x.source) ], target)
    actual = SegmentedFiniteFunction.from_list(fs).coproduct(x)

    assert actual == expected

def test_indexed_tensor(fs):
    pass

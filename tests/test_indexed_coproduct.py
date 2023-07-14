from yarrow import *

from hypothesis import given
import hypothesis.strategies as st
from tests.strategies import *

@given(c=indexed_coproducts())
def test_indexed_coproduct_roundtrip(c):
    assert c == IndexedCoproduct.from_list(c.target, list(c))

@given(fsx=common_targets())
def test_indexed_coproduct_roundtrip_functions(fsx):
    fs, x = fsx
    target = 0 if len(fs) == 0 else fs[0].target
    assert fs == list(IndexedCoproduct.from_list(target, fs))

@given(fsx=common_targets())
def test_indexed_coproduct_map(fsx):
    # a bunch of finite functions "fs" with common target.
    fs, x = fsx
    target = 0 if len(fs) == 0 else fs[0].target

    expected_sources = FiniteFunction(None, [len(fs[x(i)]) for i in range(0, x.source) ])
    expected_values = FiniteFunction.coproduct_list([ fs[x(i)] for i in range(0, x.source) ], target)

    c = IndexedCoproduct.from_list(target, fs)
    assert len(c) == len(fs)

    d = c.map(x)
    assert len(d) == x.source

    assert d.sources == expected_sources
    assert d.values == expected_values

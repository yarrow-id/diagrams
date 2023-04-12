# Vector program tests
import numpy as np
from yarrow.array import numpy

from hypothesis import given
import hypothesis.strategies as st
from tests.strategies import *

# A "run" of length N being e.g. 0 1 2 3 4 ... N
_MAX_RUN_LENGTH = 128
_MAX_RUNS = 128

# A non-vectorised implementation of segmented_arange
def _slow_segmented_arange(x):
    x = np.array(x)

    N = np.sum(x) # how many values to make?
    r = np.zeros(N, dtype=x.dtype) # allocate

    k = 0
    # for each size s,
    for i in range(0, len(x)):
        size = x[i]
        # fill result with a run 0, 1, ..., s
        for j in range(0, size):
            r[k] = j
            k += 1

    return r

@given(
    x=st.lists(st.integers(min_value=0, max_value=_MAX_RUN_LENGTH), min_size=0, max_size=_MAX_RUNS)
)
def test_segmented_arange(x):
    """ Ensure the 'segmented_arange' vector program outputs runs like 0, 1, 2, 0, 1, 2, 3, 4, ... """
    # We're returning an array of size MAX_VALUE * MAX_SIZE, so keep it smallish!
    x = np.array(x, dtype=int)
    N = np.sum(x)
    a = numpy.segmented_arange(x)

    # Check we got the expected number of elements
    assert len(a) == N
    assert np.all(slow_arange(x) == a)

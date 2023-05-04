import numpy as np

def monotonic(x, strict=False):
    if len(x) <= 1:
        return True # arrays of length <= 1 are trivially sorted

    if strict:
        return np.all(x[:-1] < x[1:])
    return np.all(x[:-1] <= x[1:])

# return true if s sorts by f.
def sorts(s, f, strict=False):
    y = s >> f
    return monotonic(y.table)

# check if an array is of the form
#   [ 0, 1, 2, ..., N₀ | 0 1 2 ... N₁ | ... ]
def is_segmented_arange(x):
    # empty array is trivially segmented
    if len(x) == 0:
        return True

    # all differences should be exactly 1, except where x = 0.
    d = x[1:] - x[:-1]
    z = (x == 0)[1:]

    # either difference is 1, or it's <= 0 at the start of a run.
    return np.all((d == 1) | (z & (d <= 0)))

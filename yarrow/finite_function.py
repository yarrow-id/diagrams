import numpy as np

DTYPE=int

class FiniteFunction:
    def __init__(self, target, table, dtype=DTYPE):
        self.table = np.array(table, dtype=dtype)
        self.source = len(table)
        self.target = target

        assert len(self.table.shape) == 1 # ensure 1D array
        assert self.source >= 0
        if self.source > 0:
            assert self.target >= 0
            assert self.target > np.max(table)
    
    def __str__(self):
        return f'{self.table} : {self.source} → {self.target}'

    def __repr__(self):
        return f'FiniteFunction({self.target}, {self.table})'

    @property
    def type(f):
        return f.source, f.target

    ################################################################################
    # FiniteFunction forms a category

    @staticmethod
    def identity(n):
        assert n >= 0
        return FiniteFunction(n, np.arange(0, n, dtype=DTYPE))

    # Compute (f ; g), i.e., the function x → g(f(x))
    def compose(f, g):
        if f.target != g.source:
            raise ValueError("Can't compose FiniteFunction {f} with {g}: f.source != g.target")

        source = f.source
        target = g.target
        # here we use numpy's indexing to compute the composition in parallel
        table = g.table[f.table]

        return FiniteFunction(target, table)

    def __rshift__(f, g):
        return f.compose(g)

    # We can compare functions for equality in a reasonable way: by just
    # comparing elements.
    # This is basically because FinFun is skeletal, so we don't need to check
    # "up to isomorphism".
    def __eq__(f, g):
        return f.source == g.source \
           and f.target == g.target \
           and np.all(f.table) == np.all(g.table)

    ################################################################################
    # FiniteFunction has initial objects and coproducts
    @staticmethod
    def initial(b, dtype=DTYPE):
        return FiniteFunction(b, np.zeros(0, dtype=DTYPE))

    def inj0(a, b):
        table = np.arange(0, a, dtype=DTYPE)
        return FiniteFunction(a + b, table)

    def inj1(a, b):
        table = np.arange(a, a + b, dtype=DTYPE)
        return FiniteFunction(a + b, table)

    def coproduct(f, g):
        assert f.target == g.target
        target = f.target
        table = np.concatenate([f.table, g.table])
        return FiniteFunction(target, table)

    def __add__(f, g):
        return f.coproduct(g)

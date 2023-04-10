import yarrow.array.numpy as numpy

DTYPE='int64'

class AbstractFiniteFunction:
    """
    Define a class of finite functions parametrised over the underlying array type.
    This class assumes there is an cls._Array member which implements array primitives.
    For example, cls._Array.max() should compute the max of an array.
    This way, we can allow for numpy/GPU arrays transparently.
    """
    def __init__(self, target, table, dtype=DTYPE):
        # _Array is the "array functions module"
        # It lets us parametrise AbstractFiniteFunction by a module like "numpy".
        self.table = type(self)._Array.array(table, dtype=dtype)
        self.target = target

        assert len(self.table.shape) == 1 # ensure 1D array
        assert self.source >= 0
        if self.source > 0:
            assert self.target >= 0
            assert self.target > type(self)._Array.max(table)

    @property
    def source(self):
        return len(self.table)

    def __str__(self):
        return f'{self.table} : {self.source} → {self.target}'

    def __repr__(self):
        return f'FiniteFunction({self.target}, {self.table})'

    def __call__(self, i: int):
        if i >= self.source:
            raise ValueError("Calling {self} with {i} >= source {self.source}")
        return self.table[i]

    @property
    def type(f):
        return f.source, f.target

    ################################################################################
    # FiniteFunction forms a category

    @classmethod
    def identity(cls, n):
        assert n >= 0
        return FiniteFunction(n, cls._Array.arange(0, n, dtype=DTYPE))

    # Compute (f ; g), i.e., the function x → g(f(x))
    def compose(f, g):
        if f.target != g.source:
            raise ValueError("Can't compose FiniteFunction {f} with {g}: f.source != g.target")

        source = f.source
        target = g.target
        # Use array indexing to compute composition in parallel (if applicable
        # cls._Array backend is used)
        table = g.table[f.table]

        return type(f)(target, table)

    def __rshift__(f, g):
        return f.compose(g)

    # We can compare functions for equality in a reasonable way: by just
    # comparing elements.
    # This is basically because FinFun is skeletal, so we don't need to check
    # "up to isomorphism".
    def __eq__(f, g):
        return f.source == g.source \
           and f.target == g.target \
           and f._Array.all(f.table == g.table)

    ################################################################################
    # FiniteFunction has initial objects and coproducts
    @classmethod
    def initial(cls, b, dtype=DTYPE):
        return cls(b, cls._Array.zeros(0, dtype=DTYPE))

    @classmethod
    def inj0(cls, a, b):
        table = cls._Array.arange(0, a, dtype=DTYPE)
        return cls(a + b, table)

    @classmethod
    def inj1(cls, a, b):
        table = cls._Array.arange(a, a + b, dtype=DTYPE)
        return cls(a + b, table)

    def coproduct(f, g):
        assert f.target == g.target
        target = f.target
        table = type(f)._Array.concatenate([f.table, g.table])
        return type(f)(target, table)

    def __add__(f, g):
        return f.coproduct(g)

    ################################################################################
    # FiniteFunction as a strict symmetric monoidal category
    @staticmethod
    def unit():
        return 0

    def tensor(f, g):
        # The tensor (f @ g) is the same as (f;ι₀) + (g;ι₁)
        # however, we compute it directly for the sake of efficiency
        T = type(f)
        table = T._Array.concatenate([f.table, g.table + f.target])
        return T(f.target + g.target, table)

    def __matmul__(f, g):
        return f.tensor(g)

    @classmethod
    def twist(cls, a, b):
        # Read a permutation as the array whose ith position denotes "where to send" value i.
        # e.g., twist_{2, 3} = [3 4 0 1 2]
        #       twist_{2, 1} = [1 2 0]
        #       twist_{0, 2} = [0 1]
        table = cls._Array.concatenate([b + cls._Array.arange(0, a), cls._Array.arange(0, b)])
        return cls(a + b, table)

    ################################################################################
    # Coequalizers for FiniteFunction
    def coequalizer(f, g):
        """
        Given finite functions    f, g : A → B,
        return the *coequalizer*  q    : B → Q
        which is the unique arrow such that  f >> q = g >> q
        """

        if f.type != g.type:
            raise ValueError(
                f"cannot coequalize arrows {f} and {g} of different types: {f.type} != {g.type}")

        # connected_components returns:
        #   Q:        number of components
        #   q: B → Q  map assigning vertices to their component
        # For the latter we have that
        #   * if f.table[i] == g.table[i]
        #   * then q[f.table[i]] == q[g.table[i]]
        # NOTE: we pass f.target so the size of the sparse adjacency matrix
        # representing the graph can be computed efficiently; otherwise we'd
        # have to take a max() of each table.
        # Q: number of connected components
        T = type(f)
        Q, q = T._Array.connected_components(f.table, g.table, f.target)
        return T(Q, q)

    ################################################################################
    # FiniteFunction also has cartesian structure which is useful
    @classmethod
    def terminal(cls, a, dtype=DTYPE):
        return cls(1, cls._Array.zeros(a, dtype=DTYPE))

    def argsort(f: 'AbstractFiniteFunction'):
        """
        Given a finite function                     f : A → B
        Return the *stable* sorting permutation     p : A → A
        such that                                   p >> f  is monotonic.
        """
        return type(f)(f.source, f._Array.argsort(f.table))

class FiniteFunction(AbstractFiniteFunction):
    """ Finite functions backed by numpy arrays """
    _Array = numpy

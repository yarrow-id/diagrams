"""An implementation of finite functions as arrays.
All yarrow datastructures are ultimately built from finite functions.
For an overview, see :cite:t:`dpafsd`, Section 2.2.2.

Finite functions can be thought of as a thin wrapper around integer arrays whose
elements are within a specified range.
Here's an example of contructing a finite function:

>>> print(FiniteFunction(3, [0, 1, 2, 0]))
[0 1 2 0] : 4 → 3

Mathematically, this represents a function from the set of 4 elements to the set
of 3 elements, and so its "type" is ``4 → 3``.

There are several constructors for finite functions corresponding to useful
morphisms in category theory.
For example, the ``identity`` map is like numpy's ``arange``:

>>> print(FiniteFunction.identity(5))
[0 1 2 3 4] : 5 → 5

and the ``terminal`` map is an array of zeroes:

>>> print(FiniteFunction.terminal(5))
[0 0 0 0 0] : 5 → 1

Finite functions form a *symmetric monoidal category*.
They can be composed sequentially:

>>> print(FiniteFunction.identity(5) >> FiniteFunction.terminal(5))
[0 0 0 0 0] : 5 → 5

And in parallel:

>>> FiniteFunction.identity(5) @ FiniteFunction.terminal(5)
FiniteFunction(6, [0 1 2 3 4 5 5 5 5 5])
"""

from typing import List
import yarrow.array.numpy as numpy

DTYPE='int64'

class AbstractFiniteFunction:
    """
    Finite functions parametrised over the underlying array type (the "backend").
    This implementation assumes there is a cls._Array member implementing various primitives.
    For example, cls._Array.sum() should compute the sum of an array.
    """
    def __init__(self, target, table, dtype=DTYPE):
        # TODO: this constructor is too complicated; it should be simplified.
        # _Array is the "array functions module"
        # It lets us parametrise AbstractFiniteFunction by a module like "numpy".
        Array = type(self)._Array
        if type(table) == Array.Type:
           self.table = table
        else:
            self.table = Array.array(table, dtype=dtype)

        self.target = target

        assert len(self.table.shape) == 1 # ensure 1D array
        assert self.source >= 0
        if self.source > 0 and self.target is not None:
            assert self.target >= 0
            assert self.target > Array.max(table)

    @property
    def source(self):
        """The source (aka "domain") of this finite function"""
        return len(self.table)

    def __len__(self):
        """Same as self.source.
        Sometimes this is clearer when thinking of a finite function as an array.
        """
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
        """Get the source and target of this finite function.

        Returns:
            tuple: (f.source, f.target)
        """
        return f.source, f.target

    ################################################################################
    # FiniteFunction forms a category

    @classmethod
    def identity(cls, n: int):
        """Return the identity finite function of type n → n.
        Args:
            n(int): The object of which to return the identity map

        Returns:
            AbstractFiniteFunction: Identity map at n
        """
        assert n >= 0
        return cls(n, cls._Array.arange(0, n, dtype=DTYPE))

    # Compute (f ; g), i.e., the function x → g(f(x))
    def compose(f: 'AbstractFiniteFunction', g: 'AbstractFiniteFunction'):
        """Compose this finite function with another

        Args:
            g: A FiniteFunction for which self.target == g.source

        Returns:
            The composition f ; g.

        Raises:
            ValueError: if self.target != g.source
        """
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
        """Compute the initial map ``? : 0 → b``"""
        return cls(b, cls._Array.zeros(0, dtype=DTYPE))

    @classmethod
    def inj0(cls, a, b):
        """Compute the injection ``ι₀ : a → a + b``"""
        table = cls._Array.arange(0, a, dtype=DTYPE)
        return cls(a + b, table)

    @classmethod
    def inj1(cls, a, b):
        """Compute the injection ``ι₁ : b → a + b``"""
        table = cls._Array.arange(a, a + b, dtype=DTYPE)
        return cls(a + b, table)

    def inject0(f, b):
        """
        Directly compute (f ; ι₀) instead of by composition.

        >>> f.inject0(b) == f >> ι₀
        """
        return type(f)(f.target + b, f.table)

    def inject1(f, a):
        """
        Directly compute (f ; ι₁) instead of by composition.

        >>> f.inject1(a) == f >> ι₁
        """
        return type(f)(a + f.target, a + f.table)

    def coproduct(f, g):
        """ Given maps ``f : A₀ → B`` and ``g : A₁ → B``
        compute the coproduct ``f.coproduct(g) : A₀ + A₁ → B``"""
        assert f.target == g.target
        assert f.table.dtype == g.table.dtype
        target = f.target
        table = type(f)._Array.concatenate([f.table, g.table])
        return type(f)(target, table)

    def __add__(f, g):
        """ Inline coproduct """
        return f.coproduct(g)

    ################################################################################
    # FiniteFunction as a strict symmetric monoidal category
    @staticmethod
    def unit():
        """ return the unit object of the category """
        return 0

    def tensor(f, g):
        """ Given maps
        ``f : A₀ → B₀`` and
        ``g : A₁ → B₁``
        compute the *tensor* product
        ``f.tensor(g) : A₀ + A₁ → B₀ + B₁``"""
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
        Given finite functions    ``f, g : A → B``,
        return the *coequalizer*  ``q    : B → Q``
        which is the unique arrow such that  ``f >> q = g >> q``
        having a unique arrow to any other such map.
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
        """ Compute the terminal map ``! : a → 1``. """
        return cls(1, cls._Array.zeros(a, dtype=DTYPE))

    ################################################################################
    # Sorting morphisms
    def argsort(f: 'AbstractFiniteFunction'):
        """
        Given a finite function                     ``f : A → B``
        Return the *stable* sorting permutation     ``p : A → A``
        such that                                   ``p >> f``  is monotonic.
        """
        return type(f)(f.source, f._Array.argsort(f.table))

    ################################################################################
    # Sequential-only methods
    @classmethod
    def coproduct_list(cls, fs: List['AbstractFiniteFunction'], target=None):
        """ Compute the coproduct of a list of finite functions with a common target """
        # NOTE: this function is not parallelized!
        if len(fs) == 0:
            return cls.initial(0 if target is None else target)

        # all targets must be equal
        assert all(f.target == g.target for f, g in zip(fs, fs[:1]))
        return cls(fs[0].target, cls._Array.concatenate([f.table for f in fs]))

    @classmethod
    def tensor_list(cls, fs: List['AbstractFiniteFunction']):
        if len(fs) == 0:
            return cls.initial(0)

        targets = cls._Array.array([f.target for f in fs])
        offsets = cls._Array.zeros(len(targets) + 1, dtype=type(fs[0].source))
        offsets[1:] = cls._Array.cumsum(targets) # exclusive scan
        table = cls._Array.concatenate([f.table + offset for f, offset in zip(fs, offsets[:-1])])
        return FiniteFunction(offsets[-1], table)


    ################################################################################
    # Finite coproducts
    def injections(s: 'AbstractFiniteFunction', a: 'AbstractFiniteFunction'):
        """
        Given a finite function ``s : N → K``
        representing the objects of the coproduct
        ``Σ_{n ∈ N} s(n)``
        whose injections have the type
        ``ι_x : s(x) → Σ_{n ∈ N} s(n)``,
        and given a finite map
        ``a : A → N``,
        compute the coproduct of injections

        .. code-block:: text

            injections(s, a) : Σ_{x ∈ A} s(x) → Σ_{n ∈ N} s(n)
            injections(s, a) = Σ_{x ∈ A} ι_a(x)

        So that ``injections(s, id) == id``

        Note that when a is a permutation,
        injections(s, a) is a "blockwise" version of that permutation with block
        sizes equal to s.
        """
        # segment pointers
        Array = a._Array

        # cumsum is inclusive, we need exclusive so we just allocate 1 more space.
        p = Array.zeros(s.source + 1, dtype=Array.DEFAULT_DTYPE)
        p[1:] = Array.cumsum(s.table)

        k = a >> s # avoid recomputation
        r = Array.segmented_arange(k.table)
        # NOTE: p[-1] is sum(s).
        cls = type(s)
        return cls(p[-1], r + cls._Array.repeat(p[a.table], k.table))


class FiniteFunction(AbstractFiniteFunction):
    """ Finite functions backed by numpy arrays """
    _Array = numpy

def argsort(f: AbstractFiniteFunction):
    """ Applies a stable 'argsort' to the underlying array of a finite function.
    When that finite function is a permutation, this inverts it.
    """
    return type(f)(f.source, f._Array.argsort(f.table))

def bincount(f: AbstractFiniteFunction):
    """ bincount the underlying array of a finite function

    Args:
        f: A finite function of type ``A → B``

    Returns:
        AbstractFiniteFunction: A finite function of type ``B → A+1``

    """
    # the bincount of an array
    #   f : A → B
    # is a finite function
    #   g : B → A+1
    # where
    #   g(b) = |{b . ∃a. f(a) = b}|
    return type(f)(len(f)+1, f._Array.bincount(f.table, minlength=f.target))

""" Finite coproducts for Finite Functions.
They're used to parallelize several operations like the
:py:meth:`yarrow.functor.functor.Functor.map_objects` method.
"""
from yarrow.finite_function import *
from dataclasses import dataclass
import yarrow.array.numpy as numpy

@dataclass
class AbstractIndexedCoproduct:
    """ A finite coproduct of finite functions.
    You can think of it simply as a segmented array.
    Categorically, it represents a finite coproduct::

        Σ_{i ∈ N} f_i : s(f_i) → Y

    as a pair of maps::

        sources: N            → Nat     (target is natural numbers)
        values : sum(sources) → Σ₀
    """
    # sources: an array of segment sizes (note: not ptrs)
    sources: AbstractFiniteFunction

    # values: the values of the coproduct
    values: AbstractFiniteFunction

    def __post_init__(self):
        # TODO FIXME: make this type derivable from AbstractFiniteFunction so we
        # don't need to have one version for each backend?
        self._Fun = type(self.sources)
        self._Array = self._Fun._Array

        # we always ignore the target of sources; this ensures
        # roundtrippability.
        assert self.sources.target is None
        assert type(self.values) == self._Fun
        assert len(self.values) == self._Array.sum(self.sources.table)

    @property
    def target(self):
        return self.values.target

    def __len__(self):
        """ return the number of finite functions in the coproduct """
        return len(self.sources)

    @classmethod
    def from_list(cls, target, fs: List['AbstractFiniteFunction']):
        """ Create an `AbstractIndexedCoproduct` from a list of :py:class:`AbstractFiniteFunction` """
        assert all(target == f.target for f in fs)
        return cls(
            sources=cls._Fun(None, [len(f) for f in fs], dtype=int),
            values=cls._Fun.coproduct_list(fs, target=target))

    def __iter__(self):
        """ Yield an iterator of the constituent finite functions

        >>> list(AbstractIndexedCoproduct.from_list(fs)) == fs
        True
        """
        N     = len(self.sources)

        # Compute source pointers
        s_ptr = self._Array.zeros(N+1, dtype=self.sources.table.dtype)
        s_ptr[1:] = self._Array.cumsum(self.sources.table)

        for i in range(0, N):
            yield self._Fun(self.target, self.values.table[s_ptr[i]:s_ptr[i+1]])

    def map(self, x: AbstractFiniteFunction):
        """ Given an :py:class:`AbstractIndexedCoproduct` of finite functions::

            Σ_{i ∈ X} f_i : Σ_{i ∈ X} A_i → B

        and a finite function::

            x : W → X

        return a new :py:class:`AbstractIndexedCoproduct` representing::

            Σ_{i ∈ X} f_{x(i)} : Σ_{i ∈ W} A_{x(i)} → B
        """
        return type(self)(
            sources = x >> self.sources,
            values = self.coproduct(x))

    def coproduct(self, x: AbstractFiniteFunction) -> AbstractFiniteFunction:
        """Like ``map`` but only computes the ``values`` array of an AbstractIndexedCoproduct"""
        assert x.target == len(self.sources)
        return self.sources.injections(x) >> self.values



@dataclass
class AbstractSegmentedFiniteFunction:
    """ An AbstractSegmentedFiniteFunction encodes a *tensoring* of finite functions.
    This means we have to include an array of *targets* as well.

    ..warning::
        Deprecated
    """
    # sizes of each of the N segments
    # sources : N → Nat
    sources: AbstractFiniteFunction

    # Targets of each finite function.
    # This is required if we want to tensor
    # targets : N → Nat
    targets: AbstractFiniteFunction

    # values of all segments, flattened
    # value : Σ_{i ∈ N} size(i) → Nat
    values: AbstractFiniteFunction

    def __post_init__(self):
        cls = type(self)
        assert self.sources._Array == cls._Array
        assert self.targets._Array == cls._Array
        assert self.values._Array == cls._Array

        # Check that values : Σ_{i ∈ N} n(i)
        assert self._Array.sum(self.sources.table) == len(self.values)

        # lengths of sources and targets arrays are the same
        assert len(self.sources) == len(self.targets)

        if len(self.targets) == 0:
            self._is_coproduct = True
        else:
            self._is_coproduct = \
                    self._Array.all(self.targets.table[:-1] == self.targets.table[1:])

    # return the number of segments
    def __len__(self):
        return len(self.sources)

    @classmethod
    def from_list(cls, fs: List['AbstractFiniteFunction']):
        """ Create a SegmentedFiniteFunction from a list of morphisms """
        # TODO: tidy up. do 1 iteration instead of 3
        sources = cls._Array.array([ f.source for f in fs ])
        targets = cls._Array.array([ f.target for f in fs ])

        if len(fs) == 0:
            max_source = 0
            max_target = 0
            values = cls._Array.zeros(0)
        else:
            max_source = cls._Array.max(sources) + 1
            max_target = cls._Array.max(targets) + 1
            values  = cls._Array.concatenate([ f.table for f in fs])

        return cls(
            sources = cls._Fun(max_source, sources),
            targets = cls._Fun(max_target, targets),
            values  = cls._Fun(None, values))

    def __iter__(self):
        Fun   = type(self.sources)
        Array = Fun._Array
        N     = len(self.sources)

        s_ptr = Array.zeros(N+1, dtype=self.sources.table.dtype)
        s_ptr[1:] = Array.cumsum(self.sources.table)

        for i in range(0, N):
            yield Fun(self.targets(i), self.values.table[s_ptr[i]:s_ptr[i+1]])

    @property
    def N(self):
        # number of segments in the array
        return self.sources.source

    def slice(self, x: AbstractFiniteFunction):
        # check indexing function won't go out of bounds
        assert x.target == self.N
        return self.sources.injections(x) >> self.values

    # Since values is the concatenation of
    # finite functions F_i : size(i) → Nat,
    # i.e.,
    #   values = F_0 + F_1 + ... + F_{N-1}
    # we have
    #   ι_x ; value = F_i
    def coproduct(self, x: AbstractFiniteFunction):
        """ sff.coproduct(x) computes an x-indexed coproduct of sff """
        # check all targets are the same
        assert self._is_coproduct

        # TODO FIXME: this is a hack, and is totally broken for "empty" coproducts, which MUST have target specified!
        target = 0 if self.targets.source == 0 else self.targets(0)
        return type(x)(target, self.slice(x).table)

    def tensor(self, x: AbstractFiniteFunction):
        """ sff.coproduct(x) computes an x-indexed *tensor* product of sff """
        table = self.slice(x).table
        p = self._Array.zeros(x.source + 1, dtype='int64')
        # p[1:] = self._Array.cumsum(self.targets.table[x.table])
        p[1:] = self._Array.cumsum(self.targets.table[x.table])
        z = self._Array.repeat(p[:-1], self.sources.table[x.table])
        return type(x)(p[-1], table + z)

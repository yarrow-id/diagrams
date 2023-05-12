from yarrow.finite_function import *
from dataclasses import dataclass
import yarrow.array.numpy as numpy

@dataclass
class AbstractSegmentedFiniteFunction:
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
        assert self._Array.sum(self.sources.table) == self.values.source

        # lengths of sources and targets arrays are the same
        assert self.sources.source == self.targets.source

        if self.targets.source == 0:
            self._is_coproduct = True
        else:
            self._is_coproduct = \
                    self._Array.all(self.targets.table[:-1] == self.targets.table[1:])

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

    @property
    def N(self):
        # number of segments in the array
        return self.sources.source

    def slice(self, x: FiniteFunction):
        # check indexing function won't go out of bounds
        assert x.target == self.N
        return self.sources.injections(x) >> self.values

    # Since values is the concatenation of
    # finite functions F_i : size(i) → Nat,
    # i.e.,
    #   values = F_0 + F_1 + ... + F_{N-1}
    # we have
    #   ι_x ; value = F_i
    def coproduct(self, x: FiniteFunction):
        """ sff.coproduct(x) computes an x-indexed coproduct of sff """
        # check all targets are the same
        assert self._is_coproduct
        target = 0 if self.targets.source == 0 else self.targets(0)
        return FiniteFunction(target, self.slice(x).table)

    def tensor(self, x: FiniteFunction):
        """ sff.coproduct(x) computes an x-indexed *tensor* product of sff """
        table = self.slice(x).table
        p = self._Array.zeros(x.source + 1, dtype='int64')
        # p[1:] = self._Array.cumsum(self.targets.table[x.table])
        p[1:] = self._Array.cumsum(self.targets.table[x.table])
        z = self._Array.repeat(p[:-1], self.sources.table[x.table])
        return FiniteFunction(p[-1], table + z)


class SegmentedFiniteFunction(AbstractSegmentedFiniteFunction):
    _Array = numpy
    _Fun   = FiniteFunction

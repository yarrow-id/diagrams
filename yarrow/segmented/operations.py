from dataclasses import dataclass
from yarrow.finite_function import AbstractFiniteFunction
from yarrow.segmented.finite_function import AbstractSegmentedFiniteFunction

def _is_valid(ops: 'Operations'):
    """ Check if a tensoring of operations has correct types """
    N = ops.xn.source
    return len(ops.s_type.sources) == N and \
           len(ops.s_type.targets) == N and \
           len(ops.t_type.sources) == N and \
           len(ops.t_type.targets) == N

@dataclass
class Operations:
    """ A flat array representation of a sequence of (typed) operations.
    Since polymorphic operations have variable types, in order to get a
    completely flat representation, we need to store them in *segmented arrays*.
    The Operations type is therefore a 3-tuple:

    Operation labels

      xn         : N            → Σ₁

    Source types

      s_type
          sources: N            → K₀
          values : sum(sources) → Σ₀      (= max(targets))
          targets: N            → Σ₀      (= const Σ₀+1)

    Target types

      t_type
          sources: N            → K₁
          values : sum(sources) → Σ₀      (= max(targets))
          targets: N            → Σ₀      (= const Σ₀+1)

    The "sources" arrays of s_type and t_type store
    """
    xn: AbstractFiniteFunction
    s_type: AbstractSegmentedFiniteFunction
    t_type: AbstractSegmentedFiniteFunction

    def __post_init__(self):
        assert _is_valid(self)
        # check types of finite functions, segmented finite functions are equal
        assert self.s_type._Fun == type(self.xn)
        assert type(self.s_type) == type(self.t_type)

    # return the number of operations
    def __len__(self):
        return len(self.xn)

    def __iter__(self):
        yield from zip(self.xn.table, self.s_type, self.t_type, strict=True)

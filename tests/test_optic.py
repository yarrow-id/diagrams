from hypothesis import given
from dataclasses import dataclass
import hypothesis.strategies as st
from tests.strategies import *

from yarrow.functor.functor import *
from yarrow.functor.optic import *

# Assume a finite signature Σ.
#
# For each operation Σ₁
#
#   f : Ai → Aj
#
# Define residuals
#
#   M_f
#
# And two operations in Ω₁ = Σ₁*2:
#
#   fwd(f) : fwd(Ai) ● M_f → fwd(Aj)
#   rev(f) : M_f ● rev(Ai) → rev(Aj)
# 
# with fwd(f) = 2*f
#      rev(f) = 2*f+1
# 
# The types are determined by the action of fwd/rev on *objects*.

@dataclass
class FiniteOpticFunctor(FrobeniusOpticFunctor):
    """ Optic Functor whose action on generating objects is given, and whose action on operations is ... """
    # Assume categories C presented by Σ and D presented by Ω.
    # objects in Σ₀ are mapped to lists Ω₀*
    wn_fwd: SegmentedFiniteFunction  # Σ₀ → Ω₀*
    wn_rev: SegmentedFiniteFunction  # Σ₀ → Ω₀*
    # Σ₀ → Ω₁ where Ω₁ = 2 * Σ₀

    # Residuals map generating operations into some choice of object in D
    # residuals : Σ₁ → Ω₀*
    #   sources : Σ₁ → Ks
    #   targets : Σ₁ → Kt
    #   values  : Σ₁ → Ω₀
    _residuals: SegmentedFiniteFunction

    def __post_init__(self):
        assert len(self.wn_fwd) == len(self.wn_rev)
        assert self.wn_fwd.values.target == self._residuals.values.target
        assert self.wn_rev.values.target == self._residuals.values.target

    def map_fwd_objects(self, objects) -> SegmentedFiniteFunction:
        # wn_fwd.sources associates to each object i ∈ Σ₀ a size k(i).
        # Thus to get the sizes of the coproducts, we just compose objects >> self.wn_fwd.sources
        assert objects.target == len(self.wn_fwd)
        result = SegmentedFiniteFunction(
                sources=objects >> self.wn_fwd.sources,
                targets=objects >> self.wn_fwd.targets,
                values = FiniteFunction(self.wn_fwd.values.target, self.wn_fwd.coproduct(objects).table))
        return result

    def map_rev_objects(self, objects) -> SegmentedFiniteFunction:
        assert objects.target == len(self.wn_rev)
        return SegmentedFiniteFunction(
                sources=objects >> self.wn_rev.sources,
                targets=objects >> self.wn_rev.targets,
                values = FiniteFunction(self.wn_rev.values.target, self.wn_rev.coproduct(objects).table))

    def residuals(self, ops: Operations) -> SegmentedFiniteFunction:
        assert ops.xn.target == len(self._residuals)
        return SegmentedFiniteFunction(
                sources=FiniteFunction(None, (ops.xn >> self._residuals.sources).table),
                targets=FiniteFunction(None, (ops.xn >> self._residuals.targets).table),
                values=FiniteFunction(self.wn_fwd.values.target, self._residuals.coproduct(ops.xn).table))

    def map_fwd_operations(self, ops: Operations) -> Diagram:
        # Each operation f maps to the singleton diagram 
        #
        #   2*f : F(A) → F(B) ● M_f
        #
        # We could parallelize this, but it's easier to just use a loop for testing!
        omega = ops.xn.target * 2

        diagrams = []
        for (x, a, b), M in zip(ops, self.residuals(ops)):
            FA = self.map_fwd_objects(a).values
            FB = self.map_fwd_objects(b).values
            xn = FiniteFunction.singleton(2*x, omega)
            diagrams.append(Diagram.singleton(FA, FB + M, xn))

        wn = None 
        xn = None
        if len(diagrams) == 0:
            wn = FiniteFunction(self.wn_fwd.values.target, [])
            xn = FiniteFunction(ops.xn.target, [])

        return Diagram.tensor_list(diagrams, wn=wn, xn=xn), FiniteFunction(None, [ len(d.type[1]) for d in diagrams ])

    def map_rev_operations(self, ops: Operations) -> Diagram:
        omega = ops.xn.target * 2

        diagrams = []
        for (x, a, b), M in zip(ops, self.residuals(ops)):
            FA = self.map_rev_objects(a).values
            FB = self.map_rev_objects(b).values
            xn = FiniteFunction.singleton(2*x+1, omega)
            diagrams.append(Diagram.singleton(M + FB, FA, xn))

        wn = None 
        xn = None
        if len(diagrams) == 0:
            wn = FiniteFunction(self.wn_rev.values.target, [])
            xn = FiniteFunction(ops.xn.target, [])

        # return Diagram.tensor_list(diagrams, wn=wn, xn=xn)
        return Diagram.tensor_list(diagrams, wn=wn, xn=xn), FiniteFunction(None, [ len(d.type[0]) for d in diagrams ])

@st.composite
def finite_optic_functor(draw):
    sigma_0, omega_0 = draw(arrow_type())

    wn_fwd  = draw(segmented_finite_functions(N=sigma_0, Obj=omega_0))
    wn_fwd.sources.target = None # bit of a hack

    wn_rev  = draw(segmented_finite_functions(N=sigma_0, Obj=omega_0))
    wn_rev.sources.target = None # bit of a hack

    sigma_1, _ = draw(arrow_type(target=omega_0))
    residuals = draw(segmented_finite_functions(N=sigma_1, Obj=omega_0))

    return FiniteOpticFunctor(wn_fwd, wn_rev, residuals)

@st.composite
def finite_optic_functor_and_diagram(draw):
    # F : Free_Σ → Free_Ω
    #   wn_fwd    : Σ₀ → Ω₀*
    #   wn_rev    : Σ₀ → Ω₀*
    #   residuals : Σ₁ → Ω₀*

    F = draw(finite_optic_functor())
    sigma_0 = len(F.wn_fwd)
    sigma_1 = len(F._residuals)
    omega_0 = F.wn_fwd.values.target
    assert omega_0 == F.wn_rev.values.target
    assert omega_0 == F._residuals.values.target
    d = draw(diagrams(Obj=sigma_0, Arr=sigma_1))
    return F, d

@given(finite_optic_functor_and_diagram())
def test_optic_functor(Fd):
    F, d = Fd
    F.map_arrow(d)

import unittest
from hypothesis import given
import hypothesis.strategies as st
from tests.strategies import *

from yarrow.decompose.frobenius import frobenius_decomposition

# SUT
from yarrow.functor.functor import *

@given(d=diagrams())
def test_decomposition_to_operations(d):
    d = frobenius_decomposition(d)
    ops = decomposition_to_operations(d)

# Given an object L(B)
#   Bwn : B → Σ₀
# ... a half-spider
#   f  : A → B
# ... and segmented finite function representing the object map of a functor
#   sff
#       sources : Σ₀ → Nat
#       targets : Σ₀ → Nat
#       values  : sum(sources) → Ω₀
# verify that applying the functor does not raise errors.
@given(sff_f_wn=object_map_and_half_spider())
def test_apply_finite_object_map(sff_f_wn):
    sff, f, Bwn = sff_f_wn
    FBwn = apply_finite_object_map(sff, Bwn)

# Check that mapping half-spiders is natural
#   f    :   A  →   B
#   F(f) : F(A) → F(B)
# Proposition 9.6, paper v1
@given(sff_f_wn=object_map_and_half_spider())
def test_map_half_spider(sff_f_wn):
    sff, f, Bwn = sff_f_wn
    Awn = f >> Bwn # A(wn) = f ; B(wn) by naturality

    # the object map of the functor we're applying.
    # For the purposes of this test, it's finite.
    def object_map(wn):
        return apply_finite_object_map(sff, wn)

    # Compute F(B)(wn) and F(f)
    FBwn = object_map(Bwn)
    Ff = map_half_spider(FBwn, f)

    # F(A)(wn)
    FAwn = Ff >> FBwn.values

    # Holds by Proposition 9.6
    assert object_map(Awn).values == FAwn

# Ensure that applying the identity object map to an object
# L(A) (represented by wn) gives the array wn.
@given(c=diagrams())
def test_identity_object_map(c):
    wn = c.G.wn
    Fwn = identity_object_map(wn)
    assert Fwn.values == wn
    assert np.all(Fwn.sources.table == 1)
    assert np.all(Fwn.targets.table == c.G.wn.target)

# Check that the identity functor implemented as a 'Frobenius Functor' gives the
# input diagram back.
@given(c=diagrams())
def test_identity_frobenius_functor(c):
    F = FrobeniusIdentity()
    d = F.map_arrow(c)

    A, B = c.type 
    C, D = d.type

    assert A == C
    assert B == D

    # for this particular functor, we expect exact equality of diagrams.
    assert c == d

class DaggerDaggerFunctor(FrobeniusFunctor):
    """ The DaggerDagger functor maps each generating morphism

        f : A → B

    to the composition

        f ; f† ; f : A → B

    and so is identity-on-objects.
    """
    def map_objects(self, objects: AbstractFiniteFunction):
        return identity_object_map(objects)

    def map_operations(self, ops: Operations) -> Diagram:
        d = Diagram.tensor_operations(ops)
        return (d >> d.dagger() >> d)

# A more complicated test for the FrobeniusFunctor class:
# make sure that even diagrams with internal wiring are mapped correctly.
# This checks that the diagram has the expected type and number of operations.
@given(c=diagrams())
def test_dagger_dagger_functor(c):
    F = DaggerDaggerFunctor()
    d = F.map_arrow(c)

    A, B = c.type 
    C, D = d.type

    assert A == C
    assert B == D

    # d has exactly 3x as many generating operations as c.
    assert d.G.xn.source == c.G.xn.source * 3

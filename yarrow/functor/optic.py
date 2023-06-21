from yarrow.diagram import AbstractDiagram
from yarrow.functor.functor import *

# A generic base class for optics
class FrobeniusOpticFunctor(FrobeniusFunctor):

    @abstractmethod
    def map_fwd_objects(self, objects: AbstractFiniteFunction) -> SegmentedFiniteFunction:
        ...

    @abstractmethod
    def map_rev_objects(self, objects: AbstractFiniteFunction) -> SegmentedFiniteFunction:
        ...

    @abstractmethod
    def residuals(self, ops: Operations) -> SegmentedFiniteFunction:
        ...

    @abstractmethod
    def map_fwd_operations(self, ops: Operations) -> Diagram:
        ...

    @abstractmethod
    def map_rev_operations(self, ops: Operations) -> Diagram:
        ...

    ############################################################################
    # Implementation

    def map_objects(self, objects: AbstractFiniteFunction) -> SegmentedFiniteFunction:
        # interleave forward and reverse objects
        fwd = self.map_fwd_objects(objects)
        rev = self.map_rev_objects(objects)

        # this must hold because len(fwd) == len(objects).
        N = len(objects)
        assert N == len(fwd)
        assert N == len(rev)
        Fun = type(objects)

        # return the blockwise interleaving of fwd/rev
        i = Fun.cointerleave(N)
        sources = Fun(None, fwd.sources.table + rev.sources.table)

        # TODO: FIXME: make "targets" optional in SegmentedFiniteFunction; allow
        # it to also be a scalar value representing an array of the same length
        # as 'sources'. This will prevent a lot of wasted memory.
        targets = Fun(None, Fun._Array.full(len(sources), objects.target, dtype=objects.table.dtype))

        # TODO: replace this with IndexedCoproduct, and allow taking coproducts of indexed coproducts!
        both = SegmentedFiniteFunction(
            sources = fwd.sources + rev.sources,
            targets = fwd.targets + rev.targets,
            values  = fwd.values  + rev.values)

        # TODO: FIXME: replacing target of values here is a hack to work around
        # bad design of SegmentedFiniteFunction
        sigma_1 = fwd.values.target
        result = SegmentedFiniteFunction(
            sources = sources,
            values  = Fun(sigma_1, both.coproduct(i).table),
            targets = targets)
        return result

    def map_operations(self, ops: Operations) -> Diagram:
        # TODO: add diagram from notes 2023-06-12
        fwds = self.map_fwd_operations(ops)
        revs = self.map_rev_operations(ops)

        Diagram = type(fwds)
        Fun = fwds._Fun
        xn = Fun.initial(fwds.G.xn.source) >> fwds.G.xn

        # We need the sizes of each of these types in order to compute
        # interleavings.
        Afwd = self.map_fwd_objects(ops.s_type.values)
        Arev = self.map_rev_objects(ops.s_type.values)
        Bfwd = self.map_fwd_objects(ops.t_type.values)
        Brev = self.map_rev_objects(ops.t_type.values)
        Na = len(Afwd.sources) # number of source wires before functor
        Nb = len(Bfwd.sources) # number of target wires before functor

        # Get the residuals for each operation
        M = self.residuals(ops)

        # 'Internal' interleaving maps which bundle together all the Bfwd / M values
        # so we can pass all the M's to "revs".
        wn1 = Bfwd.values + M.values
        i1 = (Bfwd.sources + M.sources).injections(Fun.cointerleave(len(ops.xn)))
        i1 = Diagram.half_spider(i1, wn1, xn)

        wn2 = M.values + Brev.values
        i2 = (M.sources + Brev.sources).injections(Fun.cointerleave(len(ops.xn)))
        i2 = Diagram.half_spider(i2, wn2, xn).dagger()

        id_Bfwd = Diagram.identity(Bfwd.values, xn)
        id_Brev = Diagram.identity(Brev.values, xn)
        x = (fwds >> i1) @ id_Brev
        y = id_Bfwd @ (i2 >> revs)
        c = x >> y

        # Bend wires to make an optic.
        # Sources: fwd sources and rev targets
        # Targets: fwd targets and rev sources
        s = (Fun.inj0(len(Afwd.values), len(Brev.values)) >> c.s) + (Fun.inj1(len(Bfwd.values), len(Arev.values)) >> c.t)
        t = (Fun.inj0(len(Bfwd.values), len(Arev.values)) >> c.t) + (Fun.inj1(len(Afwd.values), len(Brev.values)) >> c.s)
        d = Diagram(s, t, c.G)

        # Finally, interleave Afwd/Arev and Bfwd/Brev so optics can be tensored.
        lhs = (Afwd.sources + Arev.sources).injections(Fun.cointerleave(Na))
        rhs = (Bfwd.sources + Brev.sources).injections(Fun.cointerleave(Nb))

        lhs = Diagram.half_spider(lhs, d.type[0], xn)
        rhs = Diagram.half_spider(rhs, d.type[1], xn).dagger()
        return lhs >> d >> rhs

def lens_fwd(ops: Operations, copy_label) -> AbstractDiagram:
    # Given a tensoring
    #       f₀ ● f₁ ● ... ● fn
    # Return the diagram representing the forward part of a lens optic, i.e.,
    #       Δ ; (f₀ ● id)  ● Δ ; (f₁ ● id) ... 
    # This is given by 
    pass

def adapt_optic(optic: Diagram, Afwd, Arev, Bfwd, Brev):
    pass

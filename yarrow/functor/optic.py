""" Functors to diagrams of optics.

This module contains a class :py:class:`FrobeniusOpticFunctor`, which can be
used to implement a functor from diagrams into a category of *optics*.

"""
from yarrow.diagram import AbstractDiagram
from yarrow.functor.functor import *

# A generic base class for optics
class FrobeniusOpticFunctor(FrobeniusFunctor):

    @abstractmethod
    def map_fwd_objects(self, objects: AbstractFiniteFunction) -> AbstractIndexedCoproduct:
        ...

    @abstractmethod
    def map_rev_objects(self, objects: AbstractFiniteFunction) -> AbstractIndexedCoproduct:
        ...

    @abstractmethod
    def residuals(self, ops: Operations) -> AbstractIndexedCoproduct:
        ...

    @abstractmethod
    def map_fwd_operations(self, ops: Operations) -> AbstractDiagram:
        ...

    @abstractmethod
    def map_rev_operations(self, ops: Operations) -> AbstractDiagram:
        ...

    ############################################################################
    # Implementation

    def map_objects(self, objects: AbstractFiniteFunction) -> AbstractIndexedCoproduct:
        """ Implements map_objects in terms of ``map_fwd_objects`` and ``map_rev_objects``. """
        # look up concrete impl. of IndexedCoproduct
        IndexedCoproduct = type(objects).IndexedCoproduct

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

        both = IndexedCoproduct(
            sources = fwd.sources + rev.sources,
            values  = fwd.values  + rev.values)

        sigma_1 = fwd.values.target
        result = IndexedCoproduct(
            sources = sources,
            values  = Fun(sigma_1, both.coproduct(i).table))
        return result

    def map_operations(self, ops: Operations) -> AbstractDiagram:
        """ Implements ``map_operations`` using ``map_fwd_operations`` and ``map_rev_operations``. """
        # TODO: add diagram from notes 2023-06-12
        fwds, fwd_coarity = self.map_fwd_operations(ops)
        revs, rev_arity   = self.map_rev_operations(ops)

        Diagram = type(fwds)
        Fun = fwds._Fun
        xn = Fun.initial(fwds.G.xn.source) >> fwds.G.xn

        # We need the sizes of each of these types in order to compute
        # both the internal and external interleavings.
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
        fwd_output_sizes = Fun(None, fwd_coarity.table - M.sources.table) + M.sources
        i1 = fwd_output_sizes.injections(Fun.cointerleave(len(ops.xn)))
        i1 = Diagram.half_spider(i1, wn1, xn)

        wn2 = M.values + Brev.values
        rev_input_sizes = M.sources + Fun(None, rev_arity.table - M.sources.table)
        i2 = rev_input_sizes.injections(Fun.cointerleave(len(ops.xn)))
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
    """ :meta hide-value: """
    # Given a tensoring
    #       f₀ ● f₁ ● ... ● fn
    # Return the diagram representing the forward part of a lens optic, i.e.,
    #       Δ ; (f₀ ● id)  ● Δ ; (f₁ ● id) ... 
    # This is given by 
    raise NotImplementedError("TODO")

def adapt_optic(optic: AbstractDiagram, Afwd, Arev, Bfwd, Brev):
    """ :meta hide-value: """
    raise NotImplementedError("TODO")

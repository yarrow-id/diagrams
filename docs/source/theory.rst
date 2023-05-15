.. _theory:

Theory
======

.. warning::
   This section is only a quick sketch (for now).
   See :cite:t:`dpafsd` for a more in-depth explanation.

Although our datastructure is called a "Diagram", it is *not* about the
pictorial representation. Yarrow is *not* a library for drawing graphs!
In fact, the :ref:`yarrow.diagram` datastructure is akin to a *graph* or a
*tree*.

The theory of how to represent string diagrams as *hypergraphs*
is covered in detail in :cite:t:`rmsms` and :cite:t:`sdrt1`.
The yarrow library accompanies the paper :cite:t:`dpafsd`, which shows how these
hypergraphs can be encoded as *cospans of bipartite multigraphs*.
The key benefit of this encoding is to allow for *data-parallel* algorithms for
diagrams.

In terms of category theory, a value of type :py:class:`AbstractDiagram` is a
*morphism of the free symmetric monoidal category presented by the signature Σ + Frob*,
where Σ is a user-supplied *monoidal signature*.

Differentiability and Optics
----------------------------

One goal of the yarrow library is to allow for large-scale diagrammatic
differentiation for use with the *differentiable polynomial circuits*
described in :cite:t:`polycirc`.

.. warning::
   Differentiability features are currently under development.
   See :cite:t:`dpafsd`, Section 10 for details.

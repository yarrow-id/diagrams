.. _backends:

Backends
========

The implementation of yarrow is *backend-agnostic*.
Since Diagrams are ultimately backed by *arrays*, a "backend" is just a choice
of array.
The default provided backend is based on *numpy*,
but in the future we intend to add a *cupy* backend.

.. warning::
   TODO: information on code structure relating to backends

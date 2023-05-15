.. _backends:

Backends
========

The implementation of yarrow is *backend-agnostic*.
Since Diagrams are ultimately backed by *arrays*, a "backend" is just a choice
of array library.
The default provided backend is based on *numpy*.
In the future we intend to add a *cupy* backend.

Most classes in yarrow are implemented as abstract base classes
whose child classes select a particular backend by setting class members.
For example, the :py:class:`yarrow.finite_function.AbstractFiniteFunction` class assumes a member
``_Array`` is set.
The child class :py:class:`yarrow.finite_function.FiniteFunction` extends
``AbstractFiniteFunction``
by setting ``_Array = yarrow.array.numpy``.

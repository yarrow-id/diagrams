"""Array backends for :ref:`yarrow.finite_function`.

Each sub-module of :ref:`yarrow.array` is an "array backend".
Array backends provide a small number of *primitive functions*
like :func:`yarrow.array.numpy.zeros` and :func:`yarrow.array.numpy.arange` .
See :ref:`yarrow.array.numpy` (the default backend) for a list.

.. warning::
   This part of the API is likely to change significantly in future releases.

.. autosummary::
    :toctree: _autosummary
    :recursive:

    yarrow.array.numpy
    yarrow.array.cupy
"""

""" Provides backend-specific shims for the array subroutines underlying the Yarrow library.
    For example, the FiniteFunction class is defined as AbstractFiniteFunction with _Array = Numpy.
    We can later define CudaFiniteFunction with _Array = Cupy.
"""

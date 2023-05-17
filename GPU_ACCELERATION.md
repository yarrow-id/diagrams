# GPU Acceleration

GPU acceleration of diagrams is still experimental.
The following classes are exposed in `yarrow.cupy`:

- `yarrow.cupy.CupyDiagram`
- `yarrow.cupy.CupyFiniteFunction`
- `yarrow.cupy.CupyBipartiteMultigraph`

Here's a usage example:

    import yarrow.cupy as yarrow
    # Create a GPU-backed identity diagram with 10,000 objects, all labeled '0'.
    # The type of operations Σ₁ is the singleton set `{●}`.
    f = CupyDiagram.identity(CupyFiniteFunction.terminal(10000), CupyFiniteFunction.initial(1))

# CuPy dependencies

To use the GPU backend, you will need some additional dependencies:

- [cupy](https://cupy.dev/) for "GPU-accelerated numpy"
- [pylibcugraph](https://docs.rapids.ai/api/cugraph/stable/api_docs/plc/pylibcugraph/)
  for GPU-accelerated connected components.

At time of writing, you will need to use CUDA 11, since
CUDA 12 packages for `cugraph` are not yet available.
See [this issue](https://github.com/rapidsai/cugraph/issues/3271)

The following incantation seems to work for installing dependencies:

    # create a conda environment called "cuda11" with python 3.10
    conda create -n cuda11 python=3.10

    # enter the environment
    conda activate cuda11

    # Install cugraph into the environment (which comes with pylibcugraph)
    conda install -c rapidsai -c numba -c conda-forge -c nvidia cugraph cudatoolkit=11.8 pylibraft rmm

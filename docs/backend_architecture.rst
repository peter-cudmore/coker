Coker backend architecture
==========================

The coker backend lowers a traced function into a compact internal graph over a
single contiguous workspace vector.

At compile time, ``create_opgraph()`` in ``src/coker/backends/coker/core.py``
walks the tape and assigns each value to a slice of that workspace. Inputs are
packed by ``InputLayer`` in C order, and outputs are unpacked by
``OutputLayer`` using the same layout.

Runtime execution is a sequence of full-vector layers:

- ``BilinearWorkspaceLayer`` applies a sparse affine/quadratic transform to the
  whole workspace. This is used for bilinear-compatible algebra such as
  ``ADD``, ``SUB``, ``NEG``, ``TRANSPOSE``, ``MUL``, ``MATMUL``, ``CROSS``, and
  ``DOT``.
- ``GenericVectorLayer`` handles non-bilinear work coordinate-by-coordinate.
  It stores one op tuple per output coordinate and propagates both values and
  tangents for ``push_forward``.

Sparse layer parameters are stored with ``dok_ndarray``-backed
``BilinearWeights`` rather than dense compiled matrices. The workspace itself
remains a dense NumPy vector, so evaluation and forward-mode autodiff operate on
simple contiguous arrays while layer metadata stays sparse.

``SparseNet`` in ``src/coker/backends/coker/ast_preprocessing.py`` executes the
layers and exposes both value evaluation and ``push_forward``. When a function
contains ``FunctionSpace`` inputs or ``None`` outputs, the backend falls back to
numpy lowering instead of building this static graph.

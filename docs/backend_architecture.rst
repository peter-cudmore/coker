Coker backend architecture
==========================

The embedded backend implementation is supplied by the separate
``coker_backend`` package. Import it before selecting the backend:

.. code-block:: python

   import coker_backend
   from coker.backends.backend import get_backend_by_name

   backend = get_backend_by_name("coker")

Importing the package registers its ``CokerBackend`` factory through Coker's
public backend registry. Coker itself does not import an embedded backend
implementation or depend on its native compiler/runtime distributions.

The backend package depends on the separately packaged ``coker_compiler`` and
``coker_runtime`` native modules. It owns their Coker-facing Python bindings,
including ordinary function lowering and the existing QP integration.

The current migration preserves the legacy artifact behavior while the
canonical bytecode compiler is introduced feature by feature. Unsupported
canonical features must fail compilation once their legacy path is retired;
they must not silently select NumPy execution. The authoritative migration
sequence and bytecode/runtime boundary are maintained with the embedded runtime
repository.

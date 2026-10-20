# Writing documentation

The public documentation uses Sphinx and reStructuredText (`.rst`). This guide
is for contributors and is not part of the public navigation. No Markdown
extension or additional documentation dependencies are required.

## Plan with author notes

1. Find the topic's page in the map below and read its `.. Author notes`.
2. Add the intended audience, problem, important facts, expected result and
   unresolved questions. Preserve substantive directions and outstanding plans.
3. Ask the assistant to edit the page from those notes. Identify wording that
   must remain verbatim and decisions that need discussion first.

For example, inside `how_to/simulation.rst`:

```rst
.. Author notes

   - Use a damped pendulum as the example.
   - Assume the reader has completed the first function tutorial.
   - Explain the state vector before introducing the dynamics callback.
   - Use CasADi and plot angle against time.
   - Decide whether this recipe needs controls or should link to path planning.
```

Author notes are reStructuredText comments and **do not appear in the built
documentation**. Keep the blank line after `.. Author notes` and indent each
bullet by three spaces. Notes can also sit under individual section headings.
Keep future EKF and PID plans in these comments rather than adding public pages
before the features exist.

Turn notes into prose that serves the page's purpose; do not publish the
planning bullets as the finished page.

## Topic map

Paths below are relative to `docs/`.

| Section | Starting-point files |
| --- | --- |
| Basics | `basics/motivation.rst`, `basics/installation.rst`, `basics/getting_started.rst`, `basics/navigation.rst`, `basics/sharp_edges.rst` |
| Tutorials | `tutorials/function_evaluation.rst`, `tutorials/parameterized_matrix.rst`, `tutorials/cumulative_products.rst`, `tutorials/conditionals.rst`, `tutorials/backends.rst` |
| How-to cookbook | `how_to/backend_interoperability.rst`, `how_to/casadi_solver_tuning.rst`, `how_to/simulation.rst`, `how_to/system_identification.rst`, `how_to/path_planning.rst`, `how_to/robot_arm.rst` |
| Architecture | `architecture/evaluation_pipeline.rst` |
| API reference | `api/index.rst` and the individual `api/*.rst` pages |

Each section's `index.rst` also accepts notes about scope, ordering, or omissions.

## Section roles

- **Basics:** orient the reader, cover installation and a first result, and
  explain sharp edges with minimal prerequisites.
- **Tutorials:** teach five concepts in order: scalar functions, reshaping a
  parameter matrix, loops with `SymbolicVector`, `if_then_else`, and backend
  selection with native function extraction. Each lesson should build on the
  preceding ones.
- **How-to cookbook:** solve a specific task for someone who knows basic
  function construction. State the task and dependencies, give short steps,
  inspect the result, and explain how to adapt it. Do not repeat the tutorials.
- **Architecture:** explain the execution model, rationale and tradeoffs,
  rather than giving a recipe or listing source modules.
- **API reference:** state precise public contracts and backend limits. Link
  to learning pages instead of repeating their walkthroughs.

Every subsection should serve its section's role. Keep the motivation page's
first-person perspective and technical meaning.

## Voice and scope

Use a professional engineering voice, direct verbs, short sentences and
British spelling. Give enough guidance to reproduce the result without
repeating prerequisites, code or conclusions. Keep headings short.

Remove marketing, unnecessary narration and release or draft bookkeeping from
public prose. Describe what an example demonstrates, not that it is complete
or executable. Keep important numerical and backend limits, but state each
limit where it matters rather than repeating caveats.

## Executable examples

Use complete imports and declare optional dependencies. Prefer the existing public
API and reusable examples over a second implementation or an internal import.
Non-plot examples use named Sphinx `testcode` groups, with `testoutput` where
output is shown. Plotting examples use `plot` directives: Sphinx executes the
displayed code and renders the resulting figures, including their assertions.
Keep the code in one place rather than duplicating it in a test or image script.
A normal `code-block` is only rendered; it is not automatically executed.
Installation shell commands are explanatory, not run by the documentation build.

When editing prose, preserve runnable code, assertions, expected output and
named Sphinx test groups. Do not imply support for an operation or backend
without checking its contract.

The docs dependency group includes Matplotlib. Both documentation CI workflows
install the CasADi and PyTorch extras, run the doctests, and build HTML with
warnings treated as errors. The examples use CPU PyTorch tensors and do not
require CUDA. Check expected numerical behaviour, not merely that an example runs.

## Build and inspect

From the repository root:

```bash
uv sync --group docs --extra casadi --extra pytorch
uv run sphinx-build -b doctest docs/ docs/_build/doctest -W --keep-going
uv run sphinx-build docs/ docs/_build/html -W --keep-going
```

Open `docs/_build/html/index.html` to inspect navigation and rendering.
Build output belongs under `docs/_build/`, not alongside the source pages.

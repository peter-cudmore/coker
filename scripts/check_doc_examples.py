"""Execute the complete Python examples in the introductory guides."""

import re
from pathlib import Path
from textwrap import dedent


def main() -> None:
    sources = (
        (Path("README.md"), r"^```python\n(.*?)^```"),
        (
            Path("docs/getting_started.rst"),
            r"^\.\. code-block:: python\n\n((?:[ \t]+[^\n]*\n|\n)+)",
        ),
    )
    for path, pattern in sources:
        namespace = {"__name__": "__main__"}
        examples = re.findall(
            pattern, path.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL
        )
        if not examples:
            raise ValueError(f"No Python examples found in {path}")
        for example in examples:
            exec(compile(dedent(example), str(path), "exec"), namespace)
        print(f"Executed {len(examples)} Python examples from {path}")


if __name__ == "__main__":
    main()

"""The User Guide's examples run, and print what the guide says they print."""

import contextlib
import io
import re
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "USER_GUIDE.md"


@pytest.mark.skipif(not GUIDE.exists(), reason="docs/USER_GUIDE.md not included")
def test_the_user_guides_examples_print_what_it_says():
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), flags=re.S)
    assert blocks
    namespace = {}
    for block in blocks:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(block, str(GUIDE), "exec"), namespace)
        printed = out.getvalue().splitlines()
        # the output a block shows: `# <output>` after a print on the same line, or the
        # comment lines right after a multi-line print
        expected = []
        lines = block.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith(("print(", "for ")) and "  # " in line:
                expected.append(line.split("  # ", 1)[1])
            elif line.startswith("# ") and i > 0 and (lines[i - 1].startswith(("print(", "    print(", "# "))):
                expected.append(line[2:])
        assert len(printed) == len(expected), (printed, expected)
        for want, got in zip(expected, printed):  # the guide may annotate after the output
            assert want.startswith(got.strip()), (want, got)

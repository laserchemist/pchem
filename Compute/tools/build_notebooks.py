"""Build the student and instructor notebooks from the single sources in src/.

Source format (percent cells, like jupytext):

    # %% [markdown]
    # Markdown text, each line starting with "# "
    # %%
    python code
    # %% [markdown] instructor     <- instructor-only cell (answers, notes)
    # %% instructor                <- instructor-only code cell

Inside code cells, a solution is marked

    ### BEGIN SOLUTION
    ...worked code...
    ### END SOLUTION

and in the student notebook it becomes "# YOUR CODE HERE" (plus a
`raise NotImplementedError` when the block sits inside a function).

    python tools/build_notebooks.py            # build both sets
    python tools/build_notebooks.py --execute  # also run the instructor set (records outputs and timing)
    python tools/build_notebooks.py --students-only   # rebuild only the student set
"""
import argparse
import glob
import os
import re
import time

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = re.compile(r"^# %%(?: \[(markdown)\])?(?: (instructor))?\s*$")


def parse(path):
    cells, kind, instr, buf = [], None, False, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))
        if text.strip():
            cells.append((kind, instr, text))

    for line in open(path, encoding="utf-8").read().split("\n"):
        m = HEADER.match(line)
        if m:
            flush()
            kind, instr, buf = (m.group(1) or "code"), bool(m.group(2)), []
        else:
            buf.append(line)
    flush()
    return cells


ASSIGN = re.compile(r"^([A-Za-z_]\w*(?:\s*,\s*\*?[A-Za-z_]\w*)*)\s*(?:\+|-|\*|/)?=(?!=)")


def _needed_names(block_lines, later_text):
    """Names assigned at the top level of a solution block that later code uses."""
    names = []
    for line in block_lines:
        m = ASSIGN.match(line)
        if m:
            for n in m.group(1).replace("*", "").split(","):
                n = n.strip()
                if n and n not in names and n not in ("_", "fig", "ax") and re.search(r"\b%s\b" % re.escape(n), later_text):
                    names.append(n)
    return names


def strip_solutions(code, later_text=""):
    """Replace solution blocks by a placeholder. For a top-level block, name the variables later code expects."""
    lines = code.split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.strip() != "### BEGIN SOLUTION":
            out.append(line); i += 1; continue
        indent = line[: len(line) - len(line.lstrip())]
        j = i + 1
        while lines[j].strip() != "### END SOLUTION":
            j += 1
        block = lines[i + 1:j]
        if indent:
            out += [indent + "# YOUR CODE HERE", indent + "raise NotImplementedError"]
        else:
            rest = "\n".join(lines[j + 1:]) + "\n" + later_text
            names = _needed_names(block, rest)
            out.append("# YOUR CODE HERE" + (": define " + ", ".join(names) if names else ""))
        i = j + 1
    return "\n".join(out)


def instructor_clean(code):
    return "\n".join(l for l in code.split("\n") if l.strip() not in ("### BEGIN SOLUTION", "### END SOLUTION"))


def build(src, student):
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = parse(src)
    for idx, (kind, instr, text) in enumerate(cells):
        later = "\n".join(t for k, _, t in cells[idx + 1:] if k == "code")
        if instr and student:
            continue
        if kind == "markdown":
            if instr:
                text = '<div style="background:#fff4e5;border-left:4px solid #e08a00;padding:8px 12px">\n\n' \
                       "**Instructor notes**\n\n" + text + "\n\n</div>"
            nb.cells.append(new_markdown_cell(text))
        else:
            nb.cells.append(new_code_cell(strip_solutions(text, later) if student else instructor_clean(text)))
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--students-only", action="store_true", help="rebuild notebooks/ only, keep the executed instructor set")
    a = ap.parse_args()
    for src in sorted(glob.glob(os.path.join(ROOT, "src", "*.py"))):
        name = os.path.splitext(os.path.basename(src))[0]
        if a.only and a.only not in name:
            continue
        for student, folder in ((True, "notebooks"), (False, "instructor")):
            if a.students_only and not student:
                continue
            nb = build(src, student)
            out = os.path.join(ROOT, folder, name + ".ipynb")
            if a.execute and not student:
                from nbclient import NotebookClient
                t = time.time()
                NotebookClient(nb, timeout=3600, kernel_name="python3",
                               resources={"metadata": {"path": os.path.join(ROOT, folder)}}).execute()
                print(f"{name}: executed in {time.time() - t:.0f} s")
            nbformat.write(nb, out)
            print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()

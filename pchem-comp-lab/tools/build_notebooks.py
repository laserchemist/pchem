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

Course-style labs (the CHEM 3376 format, with auto-checks) also use:

    # LAB                          <- first line of the source: build as a lab folder
    x = worked_answer   # SOLUTION <- student sees  x = ...
    return expr         # SOLUTION <- student sees  return ...
    # %% [markdown] answer         <- student sees "ANSWER"; instructor sees the sample answer

A lab named src/comp01_H2_O2.py is written to comp01_H2_O2/comp01_H2_O2.ipynb (students) and
instructor/comp01_H2_O2/ (solutions), each with a copy of the assets in src/comp01_H2_O2/
(the checker pchem_check.py, the answer key answers_<lab>.json, quiz .json files, data). The answer key
is encoded in the students' folder and left plain in the instructor's. Each lab folder holds exactly one notebook, because
the course's init.py finds "the" notebook as the newest .ipynb in the folder.

    python tools/build_notebooks.py            # build both sets
    python tools/build_notebooks.py --execute  # also run the instructor set (records outputs and timing)
    python tools/build_notebooks.py --students-only   # rebuild only the student set
"""
import argparse
import glob
import os
import re
import shutil
import time

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = re.compile(r"^# %%(?: \[(markdown)\])?(?: (instructor|answer))?\s*$")
SOL_LINE = re.compile(r"^(\s*)(return\b|[^=#]+?=(?!=))\s*(.*?)\s*#\s*SOLUTION\s*$")


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
            kind, instr, buf = (m.group(1) or "code"), (m.group(2) or False), []
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


def strip_solution_lines(code):
    """`lhs = rhs  # SOLUTION` -> `lhs = ...`;  `return rhs  # SOLUTION` -> `return ...`."""
    out = []
    for line in code.split("\n"):
        m = SOL_LINE.match(line)
        if m:
            lhs = m.group(2).rstrip()
            c = re.search(r"\s(#[^#'\"]*)$", m.group(3))      # keep a trailing comment such as "# hartree"
            tail = ("    " + c.group(1)) if c else ""
            out.append(f"{m.group(1)}{lhs} ...{tail}" if lhs.endswith("=") or lhs == "return" else line)
        else:
            out.append(line)
    return "\n".join(out)


def instructor_clean(code):
    code = "\n".join(l for l in code.split("\n") if l.strip() not in ("### BEGIN SOLUTION", "### END SOLUTION"))
    return re.sub(r"\s*#\s*SOLUTION\s*$", "", code, flags=re.M)


def build(src, student):
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = parse(src)
    for idx, (kind, instr, text) in enumerate(cells):
        later = "\n".join(t for k, _, t in cells[idx + 1:] if k == "code")
        if instr == "instructor" and student:
            continue
        if kind == "markdown":
            if instr == "answer":
                text = "ANSWER" if student else "**Sample answer (instructor version).** " + text
            elif instr:
                text = '<div style="background:#fff4e5;border-left:4px solid #e08a00;padding:8px 12px">\n\n' \
                       "**Instructor notes**\n\n" + text + "\n\n</div>"
            nb.cells.append(new_markdown_cell(text))
        else:
            nb.cells.append(new_code_cell(strip_solution_lines(strip_solutions(text, later)) if student
                                          else instructor_clean(text)))
    return nb


def encode_answer_keys(folder):
    """Encode every answers_*.json in a student lab folder (see pchem_check.py, "About the encoding")."""
    import importlib.util
    keys = glob.glob(os.path.join(folder, "answers_*.json"))
    if not keys:
        return
    spec = importlib.util.spec_from_file_location("pchem_check", os.path.join(folder, "pchem_check.py"))
    pchem_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pchem_check)
    for key in keys:
        pchem_check.encode_file(key)


def is_lab(src):
    with open(src, encoding="utf-8") as f:
        return f.readline().strip() == "# LAB"


def execute(nb_path, cwd):
    from nbclient import NotebookClient
    for old_log in glob.glob(os.path.join(cwd, ".*_check_log.json")):   # start the answer-check record afresh
        os.remove(old_log)
    nb = nbformat.read(nb_path, 4)
    t = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": cwd}}).execute()
    nbformat.write(nb, nb_path)
    return time.time() - t


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
        lab = is_lab(src)
        for student, folder in ((True, "notebooks"), (False, "instructor")):
            if a.students_only and not student:
                continue
            nb = build(src, student)
            if lab:                                   # course-style lab folder
                out_dir = os.path.join(ROOT, name) if student else os.path.join(ROOT, "instructor", name)
                os.makedirs(out_dir, exist_ok=True)
                assets = os.path.join(ROOT, "src", name)
                if os.path.isdir(assets):
                    shutil.copytree(assets, out_dir, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
                    if student:                       # students get the answer key encoded; instructors keep it plain
                        encode_answer_keys(out_dir)
                out = os.path.join(out_dir, name + ".ipynb")
                nbformat.write(nb, out)
                if a.execute and not student:
                    print(f"{name}: executed in {execute(out, out_dir):.0f} s")
            else:
                out = os.path.join(ROOT, folder, name + ".ipynb")
                nbformat.write(nb, out)
                if a.execute and not student:
                    print(f"{name}: executed in {execute(out, os.path.join(ROOT, folder)):.0f} s")
            print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()

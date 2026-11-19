"""
pchem_check.py
==============
Answer checks for CHEM 3376 computational lab notebooks.

Each question has a short list of tests. When a student runs a check:

  * it runs the tests on the variables in their notebook, stopping at the first one that fails;
  * a pass shows a green ✅ with one sentence on why the result matters;
  * a miss shows a ❌ with a specific hint for *that* test, and the student can fix the cell and run
    the check again as often as they like: there is no penalty for trying;
  * every attempt is recorded in a small log file next to the notebook, so the record survives a
    kernel restart, and `lab.score()` reports the score at the end.

In a lab notebook:

    from pchem_check import Checker
    from checks_comp01 import CHECKS
    lab = Checker('comp01', CHECKS)

    lab.check('3.1')      # after a question
    lab.score()           # at the end: one row per question, and the total

Writing checks (see checks_comp01.py for a full example):

    CHECKS = {
      '3.1': {
        'title': 'H2+ energy function',
        'needs': ['h2plus_energy'],                      # names that must exist and not be ...
        'tests': [
            (lambda ns: abs(ns['h2plus_energy'](1.057) + 0.602244) < 2e-5,
             "Hint shown if this test fails."),
        ],
        'ok': "Shown when every test passes.",
      },
      '2.2d': {'title': 'Discussion 2.2', 'open': 'Question 2.2 Discussion', 'min_chars': 60},
    }

An 'open' check passes when the markdown cell after the cell containing that text holds at least
`min_chars` characters of the student's own writing (it reads the *saved* notebook).

Nothing here is specific to one lab; any lab can reuse it with its own CHECKS dictionary.
"""

import datetime
import glob
import html
import json
import os

try:
    from IPython.display import HTML, display
except ImportError:                       # outside Jupyter: print instead
    HTML = None


# ── colours chosen to be readable for colour-blind students; every message also has an icon and words ──
_STYLE = {
    "pass": ("#E8F5E9", "#1B7F3B", "✅"),
    "fail": ("#FFF3E0", "#B34700", "❌"),
    "info": ("#E3F2FD", "#0B5394", "ℹ️"),
}


def _box(kind, title, body):
    bg, edge, icon = _STYLE[kind]
    return (f'<div style="background:{bg}; border-left:5px solid {edge}; padding:10px 14px; '
            f'border-radius:0 8px 8px 0; margin:6px 0; color:#1a1a1a;">'
            f'<strong>{icon} {title}</strong><br>{body}</div>')


def _show(markup, text):
    if HTML is not None:
        display(HTML(markup))
    else:
        print(text)


class Checker:
    """Runs the checks for one lab and keeps the record of attempts."""

    def __init__(self, lab_name, checks, log_file=None):
        self.lab = lab_name
        self.checks = checks
        self.log_file = log_file or f".{lab_name}_check_log.json"
        self.user = os.getenv("JUPYTERHUB_USER", "student")
        self.log = self._load()

    # ── the record ──────────────────────────────────────────────────────────
    def _load(self):
        try:
            with open(self.log_file, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) and "attempts" in data else {"attempts": []}
        except (OSError, ValueError):
            return {"attempts": []}

    def _save(self):
        self.log.update(lab=self.lab, user=self.user)
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.log, f, indent=1)
        except OSError:
            pass                                  # read-only folder: keep the record in memory only

    def _record(self, q, passed):
        self.log["attempts"].append({"q": q, "passed": bool(passed),
                                     "time": datetime.datetime.now().isoformat(timespec="seconds")})
        self._save()

    def _history(self, q):
        return [a for a in self.log["attempts"] if a["q"] == q]

    # ── running the tests ───────────────────────────────────────────────────
    @staticmethod
    def _namespace():
        try:
            return get_ipython().user_ns          # noqa: F821  (defined inside Jupyter)
        except NameError:
            import inspect
            return inspect.stack()[3].frame.f_globals

    @staticmethod
    def _open_answer(text, min_chars):
        """Length test for a written answer in the saved notebook."""
        import nbformat
        files = glob.glob("*.ipynb")
        if not files:
            return False, "I couldn't find the saved notebook in this folder."
        nb = nbformat.read(max(files, key=os.path.getmtime), nbformat.NO_CONVERT)
        for i, cell in enumerate(nb.cells):
            if text in cell.source and i + 1 < len(nb.cells):
                answer = nb.cells[i + 1].source.strip()
                if answer.upper().startswith("ANSWER") and len(answer) < 2 * min_chars:
                    answer = answer[6:].strip()
                if len(answer) >= min_chars:
                    return True, ""
                return False, (f"The answer cell under <em>{html.escape(text)}</em> has {len(answer)} characters of "
                               f"writing; it needs at least {min_chars}. Write your answer in that cell (replace "
                               f"<code>ANSWER</code>), then <b>save the notebook</b> (Ctrl/Cmd + S) and run this check "
                               f"again. The check reads the saved file.")
        return False, f"I couldn't find the question <em>{html.escape(text)}</em> in the saved notebook."

    def _run(self, q):
        """Return (passed, message) without recording anything."""
        spec = self.checks[q]
        if "open" in spec:
            ok, msg = self._open_answer(spec["open"], spec.get("min_chars", 60))
            return ok, (spec.get("ok", "Answer recorded. Your instructor will read it.") if ok else msg)
        ns = self._namespace()
        for name in spec.get("needs", []):
            if name not in ns:
                return False, (f"<code>{name}</code> doesn't exist yet. Run the cell where you define it "
                               f"(and the cells above it), then run this check again.")
            if ns[name] is Ellipsis:
                return False, f"<code>{name}</code> is still <code>...</code>: replace the <code>...</code> with your answer."
        for test, hint in spec["tests"]:
            try:
                ok = bool(test(ns))
            except Exception as err:              # a crash is feedback too
                detail = html.escape(f"{type(err).__name__}: {err}")
                if "ellipsis" in str(err).lower():
                    detail = "a <code>...</code> is still waiting for your answer somewhere in this question"
                return False, f"{hint}<br><small>Python reported: {detail}</small>"
            if not ok:
                return False, hint
        return True, spec.get("ok", "")

    # ── what students call ──────────────────────────────────────────────────
    def check(self, q):
        """Check question q, show feedback, and record the attempt."""
        if q not in self.checks:
            _show(_box("info", f"No check called '{html.escape(str(q))}'",
                       "Check the question number in this cell."), f"No check called {q!r}")
            return
        title = self.checks[q].get("title", "")
        passed, msg = self._run(q)
        self._record(q, passed)
        n = len(self._history(q))
        label = f"Question {q}" + (f" — {html.escape(title)}" if title else "")
        if passed:
            tries = "on the first try" if n == 1 else f"after {n} attempts"
            word = "answer saved" if "open" in self.checks[q] else f"correct, {tries}!"
            _show(_box("pass", f"{label}: {word}", msg), f"PASS {q} ({tries})")
        else:
            again = ("" if "open" in self.checks[q] else
                     "<br><em>Fix the cell, run it, then run this check again. There is no penalty for trying.</em>")
            _show(_box("fail", f"{label}: not yet (attempt {n})", msg + again),
                  f"NOT YET {q} (attempt {n}): {msg}")

    def score(self, show=True):
        """Show the score: one row per question, from the record of attempts. Returns (passed, total)."""
        rows, passed_total, first_try = [], 0, 0
        for q, spec in self.checks.items():
            hist = self._history(q)
            ever = any(a["passed"] for a in hist)
            first = next((i + 1 for i, a in enumerate(hist) if a["passed"]), None)
            passed_total += ever
            first_try += first == 1
            done = "✅ answered" if "open" in spec else "✅ correct"
            status = done if ever else ("❌ not yet" if hist else "— not checked")
            rows.append((q, spec.get("title", ""), status, len(hist), first or ""))
        total = len(self.checks)
        if show:
            td = "style='padding:3px 12px; text-align:{}'"
            body = "".join(f"<tr><td {td.format('left')}>{q}</td><td {td.format('left')}>{html.escape(t)}</td>"
                           f"<td {td.format('left')}>{s}</td><td {td.format('center')}>{n}</td>"
                           f"<td {td.format('center')}>{f}</td></tr>" for q, t, s, n, f in rows)
            head = "".join(f"<th {td.format(a)}>{h}</th>" for h, a in (("Question", "left"), ("Topic", "left"),
                           ("Status", "left"), ("Attempts", "center"), ("Correct on attempt", "center")))
            table = f"<table style='border-collapse:collapse'><tr>{head}</tr>{body}</table>"
            kind = "pass" if passed_total == total else "info"
            summary = (f"Score: {passed_total} of {total} questions correct ({100 * passed_total / total:.0f} %), "
                       f"{first_try} of them on the first try.")
            missing = [q for q, _, s, _, _ in rows if not s.startswith("✅")]
            todo = ("<br>Still to do: " + ", ".join(missing) + ". Go back, fix them, and run their checks."
                    if missing else "<br>Every question is correct. Well done!")
            _show(_box(kind, f"{self.lab}: {html.escape(self.user)}", summary + todo) + table,
                  summary + "\n" + "\n".join(f"{q:6s} {s:15s} attempts {n}" for q, _, s, n, _ in rows))
        return passed_total, total

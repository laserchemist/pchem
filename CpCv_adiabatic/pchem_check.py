"""
pchem_check.py
==============
Answer checks for CHEM 3376 lab notebooks, without gofer.

The answers live in a JSON answer key in the lab folder (e.g. answers_comp01.json). In the
students' copy the key is *encoded* (compressed and scrambled) so the answers can't be read by
opening the file or searching it; see "About the encoding" below. When a student runs a check:

  * it runs the question's tests on the variables in their notebook, stopping at the first one that fails;
  * a pass shows a green ✅ with one sentence on why the result matters;
  * a miss shows a ❌ with feedback for *that* mistake: first the lab's list of typical wrong answers
    for the test, then common slips (wrong sign, a missing unit conversion), then the test's hint;
  * every attempt is recorded in a small log file next to the notebook, so the record survives a
    kernel restart, and `lab.score()` reports the score at the end.

In a lab notebook:

    from pchem_check import Checker
    lab = Checker('comp01', 'answers_comp01.json')

    lab.check('3.1')      # after a question
    lab.score()           # at the end: one row per question, and the total
    lab.submit(name, lab_id='Comp01')   # the course Submit button (shared-public/lab_submit.py), with the
                                        # score and the written answers (abstract, conclusions, discussions)

The answer key (plain JSON while you write it)
----------------------------------------------
    {
      "lab": "comp01",
      "common_mistakes": [                       # optional: checked for every numeric test
        {"factor": 27.211386, "feedback": "Off by a factor of 27.2: a hartree/eV conversion is missing."}
      ],
      "questions": {
        "4.4": {
          "title": "H2+ bond energy",
          "needs": ["De_h2p"],                   # names that must exist and not still be ...
          "tests": [
            {"value": "De_h2p",                  # any Python expression in the notebook's variables
             "answer": 2.787, "tol": 0.05,       # absolute tolerance (or "rtol": relative); lists for arrays
             "hint": "Shown when the value is wrong and no typical mistake matches.",
             "wrong": [                          # typical wrong answers, tried in order
               {"answer": 16.39, "tol": 0.05, "feedback": "You left out E(H)."},
               {"when": "value > 100", "feedback": "..."}   # a condition; `value` is the test's value
             ]},
            {"assert": "len(E_h2p) == len(R_grid)",          # a condition that must be True
             "hint": "...", "wrong": [{"when": "...", "feedback": "..."}]}
          ],
          "ok": "Shown when every test passes.",
          "setup": "fit = stats.linregress(x, y)"   # optional: Python run first; tests can then use `fit`
        },
        "3.4d": {"title": "Discussion", "open": "Question 3.4 Discussion", "min_chars": 60}
      }
    }

A value that is None ("a function is missing its return"), still `...`, or an array index where a
number was expected gets its own message automatically, and so does a wrong sign. "Your value: ..."
is shown under the feedback unless the test has "show_value": false (useful when the value is a
ratio or difference made up for the test).

Tests run on a copy of the notebook's variables, so `setup` and the tests can't change the notebook.

An 'open' check passes when the markdown cell after the cell containing that text holds at least
`min_chars` characters of the student's own writing (it reads the *saved* notebook).

About the encoding
------------------
    python pchem_check.py encode answers_comp01.json     # plain -> encoded, in place (or give an output file)
    python pchem_check.py decode answers_comp01.json [plain.json]   # encoded -> plain (printed, or to a file)

The build tool (tools/build_notebooks.py) encodes the key in the students' folder automatically and
leaves the instructor's copy plain. The checker reads either form. The encoding stops casual peeking
(opening the file, grep, a text search on the hub), but it is NOT secure: the checker runs inside
the student's own kernel, so anything it can decode a determined student can decode too. The same
is true of any answer check that runs in the student's notebook, gofer included. Truly hidden
answers need grading outside the student's kernel (e.g. re-running submitted notebooks).

Nothing here is specific to one lab; any lab can reuse it with its own answer key. (A Python
dictionary of lambda tests, the earlier format, is still accepted in place of the key file.)
"""

import base64
import datetime
import glob
import hashlib
import html
import json
import math
import os
import zlib

import numpy as np

try:
    from IPython.display import HTML, display
except ImportError:                       # outside Jupyter: print instead
    HTML = None


# ── the answer key: plain JSON, or the encoded form given to students ─────────────────────
KEY_FORMAT = "pchem-key-1"


def _scramble(data, lab):
    pad = hashlib.sha256(("pchem-check:" + lab).encode()).digest()
    return bytes(b ^ pad[i % len(pad)] for i, b in enumerate(data))


def encode_key(key):
    """Encoded form (a small JSON object) of a plain answer-key dictionary."""
    lab = key.get("lab", "")
    raw = zlib.compress(json.dumps(key, ensure_ascii=False).encode("utf-8"), 9)
    return {"format": KEY_FORMAT, "lab": lab,
            "note": "Answer key for this lab's checks, encoded. Run the checks from the notebook.",
            "data": base64.b64encode(_scramble(raw, lab)).decode("ascii")}


def decode_key(blob):
    """Plain answer-key dictionary from either form."""
    if blob.get("format") != KEY_FORMAT:
        return blob
    raw = _scramble(base64.b64decode(blob["data"]), blob.get("lab", ""))
    return json.loads(zlib.decompress(raw).decode("utf-8"))


def load_key(path):
    with open(path, encoding="utf-8") as f:
        return decode_key(json.load(f))


def encode_file(path, out=None):
    """Encode a plain key file (in place unless `out` is given). An already-encoded file is left as it is."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if blob.get("format") != KEY_FORMAT:
        blob = encode_key(blob)
    with open(out or path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=1)
        f.write("\n")


# ── comparing a student's value with an answer ──────────────────────────────────────────
_DEFAULT_MISTAKES = [{"factor": -1, "feedback": "Right size, wrong sign. Check the order of a subtraction, "
                                                "or a missing minus sign."}]


def _close(value, answer, tol=None, rtol=None):
    """True if value matches answer (numbers or arrays) within tol (absolute) / rtol (relative)."""
    if isinstance(answer, str) or answer is None or isinstance(answer, bool):
        return value == answer
    try:
        v, a = np.asarray(value, dtype=float), np.asarray(answer, dtype=float)
    except (TypeError, ValueError):
        return False
    if v.shape != a.shape:
        return False
    if tol is None and rtol is None:
        rtol = 1e-9
    return bool(np.allclose(v, a, rtol=rtol or 0.0, atol=tol or 0.0))


def _scalar(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _fmt(value):
    v = _scalar(value)
    if v is not None and np.ndim(value) == 0:
        return f"{v:.6g}"
    text = repr(value)
    return text if len(text) < 80 else text[:77] + "..."


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

    def __init__(self, lab_name, key, log_file=None):
        """`key`: the answer-key file (plain or encoded JSON), or a dictionary (a key, or the older lambda checks)."""
        self.lab = lab_name
        key = load_key(key) if isinstance(key, str) else key
        if isinstance(key, dict) and "questions" in key:
            self.checks = key["questions"]
            self.mistakes = _DEFAULT_MISTAKES + key.get("common_mistakes", [])
        else:
            self.checks = key
            self.mistakes = list(_DEFAULT_MISTAKES)
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
    def _saved_cells():
        """Cells of the saved notebook in this folder (the newest .ipynb), or None."""
        import nbformat
        files = glob.glob("*.ipynb")
        if not files:
            return None
        return nbformat.read(max(files, key=os.path.getmtime), nbformat.NO_CONVERT).cells

    @staticmethod
    def _answer_after(cells, text):
        """The student's writing in the cell after the one containing `text` ('ANSWER' removed), or None."""
        for i, cell in enumerate(cells):
            if text in cell.source and i + 1 < len(cells):
                answer = cells[i + 1].source.strip()
                if answer.upper().startswith("ANSWER"):
                    answer = answer[6:].strip()
                return answer
        return None

    @classmethod
    def _open_answer(cls, text, min_chars):
        """Length test for a written answer in the saved notebook."""
        cells = cls._saved_cells()
        if cells is None:
            return False, "I couldn't find the saved notebook in this folder."
        answer = cls._answer_after(cells, text)
        if answer is None:
            return False, f"I couldn't find the question <em>{html.escape(text)}</em> in the saved notebook."
        if len(answer) >= min_chars:
            return True, ""
        return False, (f"The answer cell under <em>{html.escape(text)}</em> has {len(answer)} characters of "
                       f"writing; it needs at least {min_chars}. Write your answer in that cell (replace "
                       f"<code>ANSWER</code>), then <b>save the notebook</b> (Ctrl/Cmd + S) and run this check "
                       f"again. The check reads the saved file.")

    @staticmethod
    def _env(ns):
        env = dict(ns)
        env.setdefault("np", np)
        env.setdefault("math", math)
        return env

    @staticmethod
    def _crash(hint, err):
        detail = html.escape(f"{type(err).__name__}: {err}")
        if "ellipsis" in str(err).lower():
            detail = "a <code>...</code> is still waiting for your answer somewhere in this question"
        return f"{hint}<br><small>Python reported: {detail}</small>"

    def _run_lambda(self, test, ns):
        test, hint = test
        try:
            return bool(test(ns)), hint
        except Exception as err:                  # a crash is feedback too
            return False, self._crash(hint, err)

    def _run_test(self, test, env):
        """One key test: {"value": expr, "answer": ...} or {"assert": expr}. Returns (passed, message)."""
        hint = test.get("hint", "Not yet: look at this question again.")
        expr = test.get("value", test.get("assert"))
        try:
            value = eval(expr, env)
        except Exception as err:
            return False, self._crash(hint, err)
        code = f"<code>{html.escape(expr)}</code>"
        if value is Ellipsis:
            return False, f"{code} is still <code>...</code>: replace the <code>...</code> with your answer."
        if value is None:
            return False, (f"{code} gave <code>None</code>: nothing was returned. "
                           f"Is the function missing its <code>return</code> line?")
        if "assert" in test:
            if bool(value):
                return True, ""
        elif _close(value, test["answer"], test.get("tol"), test.get("rtol")):
            return True, ""
        feedback = self._diagnose(test, value, env)
        hide = "assert" in test or test.get("show_value") is False
        shown = "" if hide else f"<br><small>Your value: <code>{html.escape(_fmt(value))}</code></small>"
        return False, (feedback or hint) + shown

    def _diagnose(self, test, value, env):
        """Feedback for a typical wrong answer, or None."""
        for w in test.get("wrong", []):
            try:
                if "when" in w:
                    hit = bool(eval(w["when"], dict(env, value=value)))
                else:
                    hit = _close(value, w["answer"], w.get("tol"), w.get("rtol", None if "tol" in w else 1e-3))
            except Exception:
                hit = False
            if hit:
                return w["feedback"]
        if "assert" in test:
            return None
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool) \
                and isinstance(test["answer"], float) and not float(test["answer"]).is_integer():
            return ("You have a whole number where a measurement was expected. Is it an <b>index</b> "
                    "(a position in an array, as from <code>np.argmin</code>)? Use it to pick the value: "
                    "<code>x[np.argmin(y)]</code>.")
        v, a = _scalar(value), _scalar(test["answer"])
        if v is None or a is None or a == 0 or v == 0 or np.ndim(value) != 0:
            return None
        ratio = v / a
        for m in self.mistakes:
            f = m["factor"]
            if any(abs(ratio - x) < 0.01 * abs(x) for x in {f, 1 / f}):
                return m["feedback"]
        return None

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
        env = None
        if "setup" in spec:
            env = self._env(ns)
            try:
                exec(spec["setup"], env)
            except Exception as err:
                first = spec["tests"][0]
                return False, self._crash(first[1] if isinstance(first, (tuple, list)) else
                                          first.get("hint", "Not yet: look at this question again."), err)
        for test in spec["tests"]:
            if isinstance(test, (tuple, list)):                    # older format: (lambda ns: ..., hint)
                ok, msg = self._run_lambda(test, ns)
            else:
                if env is None:
                    env = self._env(ns)
                ok, msg = self._run_test(test, env)
            if not ok:
                return False, msg
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

    # ── submission ──────────────────────────────────────────────────────────
    def open_answers(self):
        """{question: the student's written answer} for every written-answer check, read from the saved notebook."""
        cells = self._saved_cells() or []
        return {q: self._answer_after(cells, spec["open"]) or ""
                for q, spec in self.checks.items() if "open" in spec}

    def submit(self, name, lab_id):
        """Show the course's Submit button (shared-public/lab_submit.py) with this lab's score and written answers.

        The written answers (abstract, conclusions, discussions) are read from the saved notebook when the
        button is clicked, so they are the latest saved version."""
        from lab_submit import submit_lab
        correct, _ = self.score(show=False)
        try:
            submit_lab(name, self.user, correct, list(self.checks), lab_id=lab_id, open_answers=self.open_answers)
        except TypeError:                         # an older lab_submit.py without open_answers=: score only
            submit_lab(name, self.user, correct, list(self.checks), lab_id=lab_id)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "encode":
        encode_file(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        print("encoded", sys.argv[3] if len(sys.argv) > 3 else sys.argv[2])
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "decode":
        text = json.dumps(load_key(sys.argv[2]), indent=1, ensure_ascii=False)
        if len(sys.argv) == 4:
            with open(sys.argv[3], "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print("decoded to", sys.argv[3])
        else:
            print(text)
    else:
        print("usage: python pchem_check.py encode KEY.json [OUT.json]  |  python pchem_check.py decode KEY.json [OUT.json]")

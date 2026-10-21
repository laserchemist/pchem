"""
pchem_entry.py
==============
Spreadsheet-style data entry for CHEM 3376 lab notebooks.

Students type measurements into a grid of boxes inside the notebook. Every
change is saved automatically to a CSV file next to the notebook, so the data
survive a kernel restart and can be re-loaded with `load_entry`.

    from pchem_entry import data_entry, load_entry, save_entry

    data_entry(columns=['Gas', 'Trial', 'dh1_cm'], filename='mydata.csv',
               text_columns=['Gas'], example='example.csv')
    t = load_entry('mydata.csv', text_columns=['Gas'])     # -> datascience Table

Nothing here is specific to one experiment; any lab can reuse it.
"""

import csv
import os

import numpy as np
import ipywidgets as widgets
from IPython.display import display
from datascience import Table


# ── reading / writing ────────────────────────────────────────────────────────
def _read_csv(filename):
    with open(filename, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _write_csv(filename, columns, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


def _to_float(s):
    s = str(s).strip()
    if s == "":
        return np.nan
    return float(s)


def load_entry(filename, text_columns=("Gas",)):
    """Read a CSV written by `data_entry` (or `save_entry`) into a Table.

    Text columns stay as strings; every other column becomes a float array
    (blank boxes become nan). Rows that are completely blank are skipped.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"No data file '{filename}' yet. Enter data in the grid above "
            f"(or click 'Load example data') and try again.")
    columns, rows = _read_csv(filename)
    rows = [r + [""] * (len(columns) - len(r)) for r in rows]
    rows = [r for r in rows if any(str(v).strip() for v in r)]
    t = Table()
    for j, c in enumerate(columns):
        vals = [r[j].strip() for r in rows]
        if c in text_columns:
            t = t.with_column(c, np.array(vals, dtype=object).astype(str))
        else:
            try:
                t = t.with_column(c, np.array([_to_float(v) for v in vals], dtype=float))
            except ValueError as err:
                bad = [v for v in vals if v and not _is_number(v)]
                raise ValueError(f"Column '{c}' has a value that is not a number: {bad}") from err
    return t


def save_entry(data, filename):
    """Save data typed into a code cell instead of the grid.

    `data` is a dict {column name: list of values} or a datascience Table.
    """
    if isinstance(data, Table):
        columns = list(data.labels)
        cols = [list(data.column(c)) for c in columns]
    else:
        columns = list(data.keys())
        cols = [list(v) for v in data.values()]
    n = max(len(c) for c in cols)
    rows = []
    for i in range(n):
        row = []
        for c in cols:
            v = c[i] if i < len(c) else ""
            if isinstance(v, float) and np.isnan(v):
                v = ""
            row.append(v)
        rows.append(row)
    _write_csv(filename, columns, rows)
    print(f"Saved {n} rows to {filename}")


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ── the grid widget ──────────────────────────────────────────────────────────
def data_entry(columns, filename, n_rows=6, text_columns=("Gas",), labels=None,
               example=None, defaults=None, box_width="120px", template=None,
               widths=None):
    """Show an editable data grid that auto-saves to `filename`.

    columns      : column names used in the CSV and in your code
    labels       : optional dict {column: header shown above the boxes (with units)}
    text_columns : columns holding words (everything else must be a number)
    example      : optional CSV path loaded by the 'Load example data' button
    defaults     : optional dict {column: value} used to pre-fill new rows
    template     : optional CSV whose rows pre-fill the grid the first time
                   (e.g. row labels such as Solution / Dilution, values blank)
    widths       : optional dict {column: CSS width} overriding box_width
    """
    labels = labels or {}
    defaults = defaults or {}
    widths = widths or {}
    rows_box = widgets.VBox()
    status = widgets.HTML()
    rows = []                                    # list of lists of Text widgets

    def make_row(values=None):
        values = values or [defaults.get(c, "") for c in columns]
        row = []
        for c, v in zip(columns, values):
            w = widgets.Text(value=str(v), placeholder="word" if c in text_columns else "number",
                             continuous_update=False,
                             layout=widgets.Layout(width=widths.get(c, box_width)))
            w.observe(lambda change: save(), names="value")
            row.append(w)
        return row

    def header():
        return widgets.HBox([widgets.HTML(f"<b>{labels.get(c, c)}</b>",
                                          layout=widgets.Layout(width=widths.get(c, box_width)))
                             for c in columns])

    def redraw():
        rows_box.children = [header()] + [widgets.HBox(r) for r in rows]

    def validate():
        problems = []
        for i, r in enumerate(rows, start=1):
            for c, w in zip(columns, r):
                v = w.value.strip()
                ok = c in text_columns or v == "" or _is_number(v)
                w.layout.border = None if ok else "2px solid #D55E00"
                if not ok:
                    problems.append(f"row {i}, <code>{c}</code> = '{v}'")
        return problems

    def save(*_):
        problems = validate()
        data = [[w.value.strip() for w in r] for r in rows]
        _write_csv(filename, columns, data)
        filled = sum(any(v for v in d) for d in data)
        if problems:
            status.value = ("<span style='color:#D55E00'>⚠️ Not a number: "
                            + "; ".join(problems) + "</span>")
        else:
            status.value = f"✅ Saved {filled} row(s) to <code>{filename}</code>"

    def set_rows(value_rows):
        rows.clear()
        for vals in value_rows:
            vals = list(vals) + [""] * (len(columns) - len(vals))
            rows.append(make_row(vals[:len(columns)]))
        redraw()
        save()

    def add_row(_):
        rows.append(make_row())
        redraw()

    def remove_row(_):
        if rows:
            rows.pop()
            redraw()
            save()

    def load_example(_):
        cols, data = _read_csv(example)
        idx = [cols.index(c) if c in cols else None for c in columns]
        set_rows([[d[i] if i is not None else "" for i in idx] for d in data])
        status.value += " (example data loaded — replace it with your own!)"

    # start from saved data if it exists, otherwise blank rows
    if os.path.exists(filename):
        cols, data = _read_csv(filename)
        idx = [cols.index(c) if c in cols else None for c in columns]
        for d in data:
            rows.append(make_row([d[i] if i is not None and i < len(d) else "" for i in idx]))
        start_msg = f"📂 Loaded saved data from <code>{filename}</code>"
    elif template and os.path.exists(template):
        cols, data = _read_csv(template)
        idx = [cols.index(c) if c in cols else None for c in columns]
        for d in data:
            rows.append(make_row([d[i] if i is not None and i < len(d) else "" for i in idx]))
        start_msg = "Type your data in the empty boxes; it saves automatically."
    else:
        start_msg = "Type your data in the boxes; it saves automatically."
    while len(rows) < n_rows:
        rows.append(make_row())
    redraw()
    status.value = start_msg

    b_add = widgets.Button(description="➕ Add row")
    b_del = widgets.Button(description="➖ Remove last row")
    b_save = widgets.Button(description="💾 Save now", button_style="primary")
    b_add.on_click(add_row)
    b_del.on_click(remove_row)
    b_save.on_click(save)
    buttons = [b_add, b_del, b_save]
    if example and os.path.exists(example):
        b_ex = widgets.Button(description="📋 Load example data", button_style="warning")
        b_ex.on_click(load_example)
        buttons.append(b_ex)

    display(widgets.VBox([rows_box, widgets.HBox(buttons), status]))

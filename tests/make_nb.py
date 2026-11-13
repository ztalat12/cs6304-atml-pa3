"""Tiny helper to build .ipynb files from (kind, source) cells, syntax-checking every code cell."""
import ast
import json


def nb(cells, path):
    out = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(cells):
        src = src.strip("\n")
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": lines, "id": f"c{i:03d}"}
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        out["cells"].append(cell)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for kind, src in cells:
        if kind == "code":
            body = "\n".join("pass" if l.lstrip().startswith(("!", "%")) else l for l in src.split("\n"))
            ast.parse(body)
    return path

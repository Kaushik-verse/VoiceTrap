"""Turn a `# %%` script into a Colab notebook (T4 GPU runtime preselected)."""

import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
cells = []
for block in src.read_text().split("# %%")[1:]:
    head, _, body = block.partition("\n")
    if head.strip() == "[markdown]":
        lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.strip().splitlines()]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.strip("\n")})
nb = {"nbformat": 4, "nbformat_minor": 0, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
out = src.with_suffix(".ipynb")
out.write_text(json.dumps(nb, indent=1))
print(out, len(cells), "cells")

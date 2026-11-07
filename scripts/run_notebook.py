"""Execute trusted project code cells without modifying the source notebook.

Logs go to stdout/stderr; scientific outputs are saved by the notebook. Use the
results viewer to display results. No Jupyter kernel or mutable notebook output
file is shared between jobs.
"""

import argparse
import json
from pathlib import Path
import os
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("notebook", type=Path)
args = parser.parse_args()
path = args.notebook.resolve()
root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
namespace = {"__name__": "__main__", "__file__": str(path)}
for i, cell in enumerate(json.loads(path.read_text())["cells"]):
    if cell["cell_type"] == "code":
        print(f"Executing {path.name} cell {i}", flush=True)
        exec(
            compile("".join(cell["source"]), f"{path.name}:cell_{i}", "exec"), namespace
        )

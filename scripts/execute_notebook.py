from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


def display(obj=None, *args, **kwargs) -> None:
    if obj is not None:
        print(obj)


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__", "display": display}
    print(f"Executing {path}")
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"-- cell {idx}")
        try:
            exec(compile(source, str(path), "exec"), namespace)
        except Exception:
            print(f"Notebook execution failed in cell {idx}: {path}")
            traceback.print_exc()
            raise
    print(f"Done {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a simple Python notebook without nbclient.")
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    execute_notebook(args.notebook)


if __name__ == "__main__":
    main()

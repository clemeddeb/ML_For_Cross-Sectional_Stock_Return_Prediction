from pathlib import Path
import json


def main():
    root = Path.cwd()
    notebook_path = root / "notebooks" / "16_robustness_analysis.ipynb"
    namespace = {"__name__": "__main__"}
    notebook = json.loads(notebook_path.read_text())
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"Executing notebook code cell {index}")
        exec(compile(source, str(notebook_path), "exec"), namespace)


if __name__ == "__main__":
    main()

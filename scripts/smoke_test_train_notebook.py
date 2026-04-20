from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "colab" / "01_train_gemma_qa.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def get_code_cell_sources(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def install_google_colab_stub() -> None:
    google_module = types.ModuleType("google")
    colab_module = types.ModuleType("google.colab")
    drive_module = types.ModuleType("google.colab.drive")

    def mount(path: str) -> None:
        print(f"[stub] google.colab.drive.mount({path!r})")

    drive_module.mount = mount
    colab_module.drive = drive_module
    google_module.colab = colab_module

    sys.modules["google"] = google_module
    sys.modules["google.colab"] = colab_module
    sys.modules["google.colab.drive"] = drive_module


def execute_cell(source: str, globals_dict: dict) -> None:
    executable_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!"):
            print(f"[skip shell magic] {stripped}")
            continue
        if stripped.startswith("%"):
            print(f"[skip notebook magic] {stripped}")
            continue
        executable_lines.append(line)
    code = "\n".join(executable_lines).strip()
    if code:
        exec(compile(code, str(NOTEBOOK_PATH), "exec"), globals_dict)


def main() -> None:
    notebook = load_notebook()
    code_cells = get_code_cell_sources(notebook)
    assert len(code_cells) >= 6, "Unexpected notebook layout"

    install_google_colab_stub()

    os.environ["QA_FINETUNE_REPO_DIR"] = str(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT / "src"))

    globals_dict = {"__name__": "__main__"}

    # Execute cells up to the data upload cell, excluding the upload itself.
    for index in [0, 1, 2, 3, 4, 5]:
        print(f"\n=== Executing code cell {index + 1} ===")
        execute_cell(code_cells[index], globals_dict)

    assert Path.cwd().resolve() == REPO_ROOT.resolve(), "Notebook did not move into the repository directory"
    assert globals_dict["MODEL_NAME"], "MODEL_NAME was not defined"
    assert globals_dict["RUN_NAME"], "RUN_NAME was not defined"
    assert "runtime_profile" in globals_dict, "runtime_profile was not created"
    assert "detect_runtime" in globals_dict, "qafinetune runtime helpers were not imported"

    print("\nNotebook smoke test passed up to the training data upload step.")


if __name__ == "__main__":
    main()

# cdft_package/utils.py

from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExecutionContext:
    input_file: Path | None = None
    input_data: str | None = None
    scratch_dir: Path | None = None
    plots_dir: Path | None = None


def create_unique_scratch_dir(base_name: str = "scratch") -> Path:
    """
    Create a unique scratch directory in the current working directory.

    Rules
    -----
    - If ./scratch does not exist → create ./scratch
    - Else create ./scratch_1, ./scratch_2, ...
    - Return Path to the created directory
    """

    root = Path.cwd()
    base_path = root / base_name

    if not base_path.exists():
        base_path.mkdir()
        return base_path

    i = 1
    while True:
        candidate = root / f"{base_name}_{i}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        i += 1


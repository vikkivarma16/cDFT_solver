# cdft_package/utils.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ExecutionContext:
    def __init__(self, input_file: Path, input_data: str, scratch_dir: Path, plots_dir: Path):
        self.input_file = input_file
        self.input_data = input_data
        self.scratch_dir = scratch_dir
        self.plots_dir = plots_dir

def get_unique_dir(base_name, root: Path = Path.cwd()):
    root = Path(root).resolve()
    base_path = root / base_name
    if not base_path.exists():
        base_path.mkdir(parents=True)
        return base_path
    i = 1
    while True:
        new_path = root / f"{base_name}_{i}"
        if not new_path.exists():
            new_path.mkdir(parents=True)
            return new_path
        i += 1

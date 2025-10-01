# scripts/create_project_structure.py
import os
from pathlib import Path


def create_structure():
    """Phase 1用のディレクトリ構造を作成"""

    dirs = [
        "core",
        "core/features",
        "core/features/implementations",
        "core/signals",
        "core/utils",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        init_file = Path(d) / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    print("✅ ディレクトリ構造を作成しました")
    for d in dirs:
        print(f"   📁 {d}/")


if __name__ == "__main__":
    create_structure()

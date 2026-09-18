"""Verify third-party dependencies before starting the bot."""
from __future__ import annotations

import importlib.util
import sys

REQUIRED = {
    "aiohttp": "aiohttp",
    "numpy": "numpy",
    "pandas": "pandas",
    "requests": "requests",
    "dotenv": "python-dotenv",
    "sklearn": "scikit-learn",
}

def main() -> int:
    missing = []
    for module, package in REQUIRED.items():
        if importlib.util.find_spec(module) is None:
            missing.append(package)
    if missing:
        print("Missing Python packages:")
        for package in missing:
            print(f"  - {package}")
        print("\nInstall them with:")
        print("  python -m pip install -r requirements.txt")
        return 1
    print("All required third-party packages are installed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

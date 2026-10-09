"""
neuromap - Root Application Launcher
Place this file in the project root: c:\\Users\\Jacob\\Documents\\neuromap\\main.py (or run_neuromap.py).
It automatically configures sys.path so that all modules in src\\neuromap\\ are discovered seamlessly.
"""

import sys
from pathlib import Path

# Automatically add src/ and src/neuromap/ to Python search path
_root = Path(__file__).resolve().parent
for _cand in [
    _root / "src" / "neuromap",
    _root / "src",
    _root,
]:
    _cand_str = str(_cand)
    if _cand.is_dir() and _cand_str not in sys.path:
        sys.path.insert(0, _cand_str)

try:
    from neuromap.main import main
except (ImportError, ModuleNotFoundError):
    from main import main

if __name__ == "__main__":
    main()

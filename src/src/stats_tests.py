"""Compatibility wrapper.

The maintained stats testing implementation is located at src/stats_tests.py.
This file is kept to avoid breaking older scripts that call src/src/stats_tests.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats_tests import main


if __name__ == "__main__":
    main()

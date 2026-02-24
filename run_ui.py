#!/usr/bin/env python3
import os
import sys

# Ensure project root is on PYTHONPATH
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.ui_server import main

if __name__ == "__main__":
    main()
